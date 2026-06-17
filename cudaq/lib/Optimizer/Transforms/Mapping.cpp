/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Device.h"
#include "cudaq/Support/Placement.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>

namespace cudaq::opt {
#define GEN_PASS_DEF_MAPPINGFUNC
#define GEN_PASS_DEF_MAPPINGPREP
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "quantum-mapper"

using namespace mlir;

#include "LoopUnrollPatterns.inc"

namespace {

constexpr StringRef mappedWireSetName("mapped_wireset");

//===----------------------------------------------------------------------===//
// Placement
//===----------------------------------------------------------------------===//

void identityPlacement(cudaq::Placement &placement) {
  for (unsigned i = 0, end = placement.getNumVirtualQubits(); i < end; ++i)
    placement.map(cudaq::Placement::VirtualQ(i), cudaq::Placement::DeviceQ(i));
}

//===----------------------------------------------------------------------===//
// Routing
//===----------------------------------------------------------------------===//

/// This class encapsulates an quake operation that uses wires with information
/// about the virtual qubits these wires correspond.
struct VirtualOp {
  mlir::Operation *op;
  SmallVector<cudaq::Placement::VirtualQ, 2> qubits;

  VirtualOp(mlir::Operation *op, ArrayRef<cudaq::Placement::VirtualQ> qubits)
      : op(op), qubits(qubits) {}
};

/// The `SabreRouter` class is modified implementation of the following paper:
/// Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem for
/// NISQ-era quantum devices." In Proceedings of the Twenty-Fourth International
/// Conference on Architectural Support for Programming Languages and Operating
/// Systems, pp. 1001-1014. 2019.
/// https://dl.acm.org/doi/pdf/10.1145/3297858.3304023
///
/// Routing starts with source operations collected during the analysis. These
/// operations form a layer, called the `frontLayer`, which is a set of
/// operations that have no unmapped predecessors. In the case of these source
/// operations, the router only needs to iterate over the front layer while
/// visiting all users of each operation. The processing of this front layer
/// will create a new front layer containing all operations that have being
/// visited the same number of times as their number of wire operands.
///
/// After processing the very first front layer, the algorithm proceeds to
/// process the newly created front layer. Once again, it processes the front
/// layer and map all operations that are compatible with the current placement,
/// i.e., one-wire operations and two-wire operations using wires that
/// correspond to qubits that are adjacently placed in the device. When an
/// operation is successfully mapped, it is removed from the front layer and all
/// its users are visited. Those users that have no unmapped predecessors are
/// added to the front layer. If the mapper cannot successfully map any
/// operation in the front layer, then it adds a swap to the circuit and tries
/// to map the front layer again. The routing process ends when the front layer
/// is empty.
///
/// Modifications from the published paper include the ability to defer
/// measurement mapping until the end, which is required for QIR Base Profile
/// programs (see the `allowMeasurementMapping` member variable).
class SabreRouter {
  using WireMap = DenseMap<Value, cudaq::Placement::VirtualQ>;
  using Swap = std::pair<cudaq::Placement::DeviceQ, cudaq::Placement::DeviceQ>;

public:
  SabreRouter(const cudaq::Device &device, WireMap &wireMap,
              cudaq::Placement &placement, unsigned extendedLayerSize,
              float extendedLayerWeight, float decayDelta,
              unsigned roundsDecayReset)
      : device(device), wireToVirtualQ(wireMap), placement(placement),
        extendedLayerSize(extendedLayerSize),
        extendedLayerWeight(extendedLayerWeight), decayDelta(decayDelta),
        roundsDecayReset(roundsDecayReset),
        phyDecay(device.getNumQubits(), 1.0), phyToWire(device.getNumQubits()),
        allowMeasurementMapping(false) {}

  /// Main entry point into SabreRouter routing algorithm
  void route(Block &block, ArrayRef<cudaq::quake::BorrowWireOp> sources);

  /// After routing, this contains the final values for all the qubits
  ArrayRef<Value> getPhyToWire() { return phyToWire; }

private:
  void visitUsers(ResultRange::user_range users,
                  SmallVectorImpl<VirtualOp> &layer,
                  SmallVectorImpl<Operation *> *incremented = nullptr);

  LogicalResult mapOperation(VirtualOp &virtOp);

  LogicalResult mapFrontLayer();

  void selectExtendedLayer();

  double computeLayerCost(ArrayRef<VirtualOp> layer);

  Swap chooseSwap();

private:
  const cudaq::Device &device;
  WireMap &wireToVirtualQ;
  cudaq::Placement &placement;

  // Parameters
  const unsigned extendedLayerSize;
  const float extendedLayerWeight;
  const float decayDelta;
  const unsigned roundsDecayReset;

  // Internal data
  SmallVector<VirtualOp> frontLayer;
  SmallVector<VirtualOp> extendedLayer;
  SmallVector<VirtualOp> measureLayer;
  llvm::SmallPtrSet<mlir::Operation *, 32> measureLayerSet;
  llvm::SmallSet<cudaq::Placement::DeviceQ, 32> involvedPhy;
  SmallVector<float> phyDecay;

  SmallVector<Value> phyToWire;

  /// Keeps track of how many times an operation was visited.
  DenseMap<Operation *, unsigned> visited;

  /// Keep track of whether or not we're in the phase that allows measurements
  /// to be mapped
  bool allowMeasurementMapping;

#ifndef NDEBUG
  /// A logger used to emit diagnostics during the maping process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif
};

void SabreRouter::visitUsers(ResultRange::user_range users,
                             SmallVectorImpl<VirtualOp> &layer,
                             SmallVectorImpl<Operation *> *incremented) {
  for (auto user : users) {
    auto [entry, created] = visited.try_emplace(user, 1);
    if (!created)
      entry->second += 1;
    if (incremented)
      incremented->push_back(user);

    if (!cudaq::quake::isSupportedMappingOperation(user)) {
      LLVM_DEBUG({
        auto *tmpOp = dyn_cast<mlir::Operation *>(user);
        logger.getOStream() << "WARNING: unsupported op: " << *tmpOp << '\n';
      });
    } else {
      auto wires = cudaq::quake::getQuantumOperands(user);
      if (entry->second == wires.size()) {
        SmallVector<cudaq::Placement::VirtualQ, 2> qubits;
        for (auto wire : wires)
          qubits.push_back(wireToVirtualQ[wire]);
        // Don't process measurements until we're ready
        if (allowMeasurementMapping ||
            !user->hasTrait<cudaq::QuantumMeasure>()) {
          layer.emplace_back(user, qubits);
        } else {
          // Add to measureLayer. Don't add duplicates.
          if (measureLayerSet.find(user) == measureLayerSet.end()) {
            measureLayer.emplace_back(user, qubits);
            measureLayerSet.insert(user);
          }
        }
      }
    }
  }
}

LogicalResult SabreRouter::mapOperation(VirtualOp &virtOp) {
  // Take the device qubits from this operation.
  SmallVector<cudaq::Placement::DeviceQ, 2> deviceQubits;
  for (auto vr : virtOp.qubits)
    deviceQubits.push_back(placement.getPhy(vr));

  // An operation cannot be mapped if it is not a measurement and uses two
  // qubits virtual qubit that are no adjacently placed.
  if (!virtOp.op->hasTrait<cudaq::QuantumMeasure>() &&
      deviceQubits.size() == 2 &&
      !device.areConnected(deviceQubits[0], deviceQubits[1]))
    return failure();

  // Rewire the operation.
  SmallVector<Value, 2> newOpWires;
  for (auto phy : deviceQubits)
    newOpWires.push_back(phyToWire[phy.index]);
  if (failed(cudaq::quake::setQuantumOperands(virtOp.op, newOpWires)))
    return failure();

  if (isa<cudaq::quake::SinkOp, cudaq::quake::ReturnWireOp>(virtOp.op))
    return success();

  // Update the mapping between device qubits and wires.
  for (auto &&[w, q] : llvm::zip_equal(
           cudaq::quake::getQuantumResults(virtOp.op), deviceQubits))
    phyToWire[q.index] = w;

  return success();
}

LogicalResult SabreRouter::mapFrontLayer() {
  bool mappedAtLeastOne = false;
  SmallVector<VirtualOp> newFrontLayer;

  LLVM_DEBUG({
    logger.startLine() << "Mapping front layer:\n";
    logger.indent();
  });
  for (auto virtOp : frontLayer) {
    LLVM_DEBUG({
      logger.startLine() << "* ";
      virtOp.op->print(logger.getOStream(),
                       OpPrintingFlags().printGenericOpForm());
    });
    if (failed(mapOperation(virtOp))) {
      LLVM_DEBUG(logger.getOStream() << " --> FAILURE\n");
      newFrontLayer.push_back(virtOp);
      for (auto vr : virtOp.qubits)
        involvedPhy.insert(placement.getPhy(vr));
      LLVM_DEBUG({
        auto phy0 = placement.getPhy(virtOp.qubits[0]);
        auto phy1 = placement.getPhy(virtOp.qubits[1]);
        logger.indent();
        logger.startLine() << "+ virtual qubits: " << virtOp.qubits[0] << ", "
                           << virtOp.qubits[1] << '\n';
        logger.startLine() << "+ device qubits: " << phy0 << ", " << phy1
                           << '\n';
        logger.unindent();
      });
      continue;
    }
    LLVM_DEBUG(logger.getOStream() << " --> SUCCESS\n");
    mappedAtLeastOne = true;
    visitUsers(virtOp.op->getUsers(), newFrontLayer);
  }
  LLVM_DEBUG(logger.unindent());
  frontLayer = std::move(newFrontLayer);
  return mappedAtLeastOne ? success() : failure();
}

void SabreRouter::selectExtendedLayer() {
  extendedLayer.clear();
  SmallVector<Operation *, 20> incremented;
  SmallVector<VirtualOp> tmpLayer = frontLayer;
  while (!tmpLayer.empty() && extendedLayer.size() < extendedLayerSize) {
    SmallVector<VirtualOp> newTmpLayer;
    for (VirtualOp &virtOp : tmpLayer)
      visitUsers(virtOp.op->getUsers(), newTmpLayer, &incremented);
    for (VirtualOp &virtOp : newTmpLayer)
      // We only add operations that can influence placement to the extended
      // frontlayer, i.e., quantum operators that use two qubits.
      if (!virtOp.op->hasTrait<cudaq::QuantumMeasure>() &&
          cudaq::quake::getQuantumOperands(virtOp.op).size() == 2)
        extendedLayer.emplace_back(virtOp);
    tmpLayer = std::move(newTmpLayer);
  }

  for (auto virtOp : incremented)
    visited[virtOp] -= 1;
}

double SabreRouter::computeLayerCost(ArrayRef<VirtualOp> layer) {
  double cost = 0.0;
  for (VirtualOp const &virtOp : layer) {
    auto phy0 = placement.getPhy(virtOp.qubits[0]);
    auto phy1 = placement.getPhy(virtOp.qubits[1]);
    cost += device.getDistance(phy0, phy1) - 1;
  }
  return cost / layer.size();
}

SabreRouter::Swap SabreRouter::chooseSwap() {
  // Obtain SWAP candidates
  SmallVector<Swap> candidates;
  for (auto phy0 : involvedPhy)
    for (auto phy1 : device.getNeighbours(phy0))
      candidates.emplace_back(phy0, phy1);

  if (extendedLayerSize)
    selectExtendedLayer();

  // Compute cost
  SmallVector<double> cost;
  for (auto [phy0, phy1] : candidates) {
    placement.swap(phy0, phy1);
    double swapCost = computeLayerCost(frontLayer);
    double maxDecay = std::max(phyDecay[phy0.index], phyDecay[phy1.index]);

    if (!extendedLayer.empty()) {
      double extendedLayerCost =
          computeLayerCost(extendedLayer) / extendedLayer.size();
      swapCost /= frontLayer.size();
      swapCost += extendedLayerWeight * extendedLayerCost;
    }

    cost.emplace_back(maxDecay * swapCost);
    placement.swap(phy0, phy1);
  }

  // Find and return the swap with minimal cost
  std::size_t minIdx = 0u;
  for (std::size_t i = 1u, end = cost.size(); i < end; ++i)
    if (cost[i] < cost[minIdx])
      minIdx = i;

  LLVM_DEBUG({
    logger.startLine() << "Choosing a swap:\n";
    logger.indent();
    logger.startLine() << "Involved device qubits:";
    for (auto phy : involvedPhy)
      logger.getOStream() << " " << phy;
    logger.getOStream() << "\n";
    logger.startLine() << "Swap candidates:\n";
    logger.indent();
    for (auto &&[qubits, c] : llvm::zip_equal(candidates, cost))
      logger.startLine() << "* " << qubits.first << ", " << qubits.second
                         << " (cost = " << c << ")\n";
    logger.getOStream() << "\n";
    logger.unindent();
    logger.startLine() << "Selected swap: " << candidates[minIdx].first << ", "
                       << candidates[minIdx].second << '\n';
    logger.unindent();
  });
  return candidates[minIdx];
}

void SabreRouter::route(Block &block,
                        ArrayRef<cudaq::quake::BorrowWireOp> sources) {
#ifndef NDEBUG
  constexpr char logLineComment[] =
      "//===-------------------------------------------===//\n";
#endif

  LLVM_DEBUG({
    logger.getOStream() << "\n";
    logger.startLine() << logLineComment;
    logger.startLine() << "Mapping front layer:\n";
    logger.indent();
    for (auto virtOp : sources)
      logger.startLine() << "* " << *virtOp << " --> SUCCESS\n";
    logger.unindent();
    logger.startLine() << logLineComment;
  });

  // The source ops can always be mapped.
  for (auto borrowWire : sources) {
    visitUsers(borrowWire->getUsers(), frontLayer);
    Value wire = borrowWire.getResult();
    auto phy = placement.getPhy(wireToVirtualQ[wire]);
    phyToWire[phy.index] = wire;
  }

  OpBuilder builder(&block, block.begin());
  auto wireType = builder.getType<cudaq::quake::WireType>();
  auto addSwap = [&](cudaq::Placement::DeviceQ q0,
                     cudaq::Placement::DeviceQ q1) {
    placement.swap(q0, q1);
    auto swap = cudaq::quake::SwapOp::create(
        builder, builder.getUnknownLoc(), TypeRange{wireType, wireType}, false,
        ValueRange{}, ValueRange{},
        ValueRange{phyToWire[q0.index], phyToWire[q1.index]},
        DenseBoolArrayAttr{});
    phyToWire[q0.index] = swap.getResult(0);
    phyToWire[q1.index] = swap.getResult(1);
  };

  std::size_t numSwapSearches = 0;
  bool done = false;
  while (!done) {
    // Once frontLayer is empty, grab everything from measureLayer and go again.
    if (frontLayer.empty()) {
      if (allowMeasurementMapping) {
        done = true;
      } else {
        allowMeasurementMapping = true;
        frontLayer = std::move(measureLayer);
      }
      continue;
    }

    LLVM_DEBUG({
      logger.getOStream() << "\n";
      logger.startLine() << logLineComment;
    });

    if (succeeded(mapFrontLayer()))
      continue;

    LLVM_DEBUG(logger.getOStream() << "\n";);

    // Add a swap
    numSwapSearches++;
    auto [phy0, phy1] = chooseSwap();
    addSwap(phy0, phy1);
    involvedPhy.clear();

    // Update decay
    if ((numSwapSearches % roundsDecayReset) == 0) {
      std::fill(phyDecay.begin(), phyDecay.end(), 1.0);
    } else {
      phyDecay[phy0.index] += decayDelta;
      phyDecay[phy1.index] += decayDelta;
    }
  }
  LLVM_DEBUG(logger.startLine() << '\n' << logLineComment << '\n';);
}

std::pair<bool, std::optional<cudaq::Device>>
deviceFromString(llvm::StringRef deviceString) {
  std::size_t deviceDim[2];
  deviceDim[0] = deviceDim[1] = 0;

  // Get device
  StringRef deviceTopoStr =
      deviceString.take_front(deviceString.find_first_of('('));

  // Trim the dimensions off of `deviceDef` if dimensions were provided in the
  // string
  if (deviceTopoStr.size() < deviceString.size())
    deviceString = deviceString.drop_front(deviceTopoStr.size());

  if (deviceTopoStr.equals_insensitive("file")) {
    StringRef deviceFilename;
    if (deviceString.consume_front("(")) {
      deviceString = deviceString.ltrim();
      if (deviceString.consume_back(")")) {
        deviceFilename = deviceString;
        // Remove any leading and trailing single quotes that may have been
        // added in order to pass files with spaces into the pass (required
        // for parsePassPipeline).
        if (deviceFilename.size() >= 2 && deviceFilename.front() == '\'' &&
            deviceFilename.back() == '\'')
          deviceFilename = deviceFilename.drop_front(1).drop_back(1);
        // Make sure the file exists before continuing
        if (!llvm::sys::fs::exists(deviceFilename)) {
          llvm::errs() << "Path " << deviceFilename << " does not exist\n";
          return std::make_pair(false, std::nullopt);
        }
      } else {
        llvm::errs() << "Missing closing ')' in device option\n";
        return std::make_pair(false, std::nullopt);
      }
    } else {
      llvm::errs() << "Filename must be provided in device option like "
                      "file(/full/path/to/device_file.txt): "
                   << deviceString << '\n';
      return std::make_pair(false, std::nullopt);
    }

    return std::make_pair(false, cudaq::Device::file(deviceFilename));
  } else {
    if (deviceString.consume_front("(")) {
      deviceString = deviceString.ltrim();

      // Parse first dimension
      deviceString.consumeInteger(/*Radix=*/10, deviceDim[0]);
      deviceString = deviceString.ltrim();

      // Parse second dimension if present
      unsigned argCount = 1;
      while (deviceString.consume_front(",")) {
        if (argCount == 2) {
          llvm::errs() << "Too many arguments provided for device\n";
          return std::make_pair(false, std::nullopt);
        }
        deviceString = deviceString.ltrim();
        deviceString.consumeInteger(/*Radix=*/10, deviceDim[1]);
        deviceString = deviceString.ltrim();
        ++argCount;
      }

      if (!deviceString.consume_front(")")) {
        llvm::errs() << "Missing closing ')' in device option\n";
        return std::make_pair(false, std::nullopt);
      }
    }

    if (deviceTopoStr == "path") {
      return std::make_pair(false, cudaq::Device::path(deviceDim[0]));
    } else if (deviceTopoStr == "ring") {
      return std::make_pair(false, cudaq::Device::ring(deviceDim[0]));
    } else if (deviceTopoStr == "star") {
      return std::make_pair(false,
                            cudaq::Device::star(deviceDim[0], deviceDim[1]));
    } else if (deviceTopoStr == "grid") {
      return std::make_pair(false,
                            cudaq::Device::grid(deviceDim[0], deviceDim[1]));
    } else if (deviceTopoStr == "bypass") {
      return std::make_pair(true, std::nullopt);
    } else {
      llvm::errs() << "Unknown device option: " << deviceTopoStr << '\n';
      return std::make_pair(false, std::nullopt);
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct MappingPrep : public cudaq::opt::impl::MappingPrepBase<MappingPrep> {
  using MappingPrepBase::MappingPrepBase;

  std::optional<cudaq::Device> deviceInstance;
  bool deviceBypass = false;

  virtual LogicalResult initialize(MLIRContext *context) override {
    std::tie(deviceBypass, deviceInstance) = deviceFromString(device);
    if (deviceInstance || deviceBypass || !nonComposable) {
      return success();
    }

    signalPassFailure();
    return failure();
  }

  /// Create an adjacency matrix attribute for a WireSetOp.
  SparseElementsAttr getAdjacencyFromDevice(cudaq::Device &d,
                                            MLIRContext *ctx) {
    int numEdges = 0;
    unsigned int qubitCardinality = static_cast<unsigned int>(d.getNumQubits());

    SmallVector<APInt, 32> edgeVector;
    for (unsigned int i = 0; i < qubitCardinality; i++) {
      auto neighbors = d.getNeighbours(cudaq::Device::Qubit(i));
      numEdges += neighbors.size();
      for (auto neighbor : neighbors) {
        edgeVector.emplace_back(64, i);
        edgeVector.emplace_back(64, neighbor.index);
      }
    }

    IntegerType boolTy = IntegerType::get(ctx, /*width=*/1);
    ShapedType tensorI1 =
        RankedTensorType::get({qubitCardinality, qubitCardinality}, boolTy);
    auto indicesType =
        RankedTensorType::get({numEdges, 2}, IntegerType::get(ctx, 64));
    auto indices = DenseIntElementsAttr::get(indicesType, edgeVector);
    auto intValue = mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get({static_cast<int64_t>(numEdges)}, boolTy),
        true);
    auto sparseInt = SparseElementsAttr::get(tensorI1, indices, intValue);

    return sparseInt;
  }

  cudaq::quake::WireSetOp insertWireSetOpForDevice(cudaq::Device &d,
                                                   ModuleOp mod) {
    if (auto wires =
            mod.lookupSymbol<cudaq::quake::WireSetOp>(mappedWireSetName))
      return wires;

    auto adjacency = getAdjacencyFromDevice(d, mod.getContext());
    OpBuilder builder(mod.getBodyRegion());
    auto wireSetOp = cudaq::quake::WireSetOp::create(
        builder, builder.getUnknownLoc(), mappedWireSetName, d.getNumQubits(),
        adjacency);
    wireSetOp.setPrivate();
    return wireSetOp;
  }

  void runOnOperation() override {
    auto mod = getOperation();

    if (deviceBypass)
      return;

    insertWireSetOpForDevice(*deviceInstance, mod);
  }
};

struct MappingFunc : public cudaq::opt::impl::MappingFuncBase<MappingFunc> {
  using MappingFuncBase::MappingFuncBase;

  bool deviceBypass = false;
  std::optional<cudaq::Device> deviceInstance;

  virtual LogicalResult initialize(MLIRContext *context) override {
    std::tie(deviceBypass, deviceInstance) = deviceFromString(device);
    if (deviceInstance || deviceBypass || !nonComposable) {
      return success();
    }

    signalPassFailure();
    return failure();
  }

  /// Add `op` and all of its users into `opsToMoveToEnd`. `op` may not be
  /// nullptr.
  void addOpAndUsersToList(Operation *op,
                           SmallVectorImpl<Operation *> &opsToMoveToEnd) {
    opsToMoveToEnd.push_back(op);
    for (auto user : op->getUsers())
      addOpAndUsersToList(user, opsToMoveToEnd);
  }

  std::optional<unsigned> getStaticReferenceQubitCount(func::FuncOp func) {
    unsigned numQubits = 0;
    bool failed = false;
    func.walk([&](cudaq::quake::AllocaOp alloca) {
      if (alloca->getParentOp() != func) {
        failed = true;
        return WalkResult::interrupt();
      }
      auto ty = alloca.getResult().getType();
      if (!cudaq::quake::isConstantQuantumRefType(ty)) {
        failed = true;
        return WalkResult::interrupt();
      }
      numQubits += cudaq::quake::getAllocationSize(ty);
      return WalkResult::advance();
    });
    if (failed)
      return std::nullopt;
    return numQubits;
  }

  std::optional<unsigned> getSingleReferenceOperandCount(ValueRange values) {
    unsigned count = 0;
    for (auto value : values) {
      auto type = value.getType();
      if (isa<cudaq::quake::RefType, cudaq::quake::WireType,
              cudaq::quake::ControlType>(type)) {
        ++count;
        continue;
      }
      if (cudaq::quake::isQuantumType(type))
        return std::nullopt;
    }
    return count;
  }

  std::optional<unsigned> getMeasurementOperandCount(ValueRange values) {
    unsigned count = 0;
    for (auto value : values) {
      auto type = value.getType();
      if (isa<cudaq::quake::RefType, cudaq::quake::WireType,
              cudaq::quake::ControlType>(type)) {
        ++count;
        continue;
      }
      if (auto veqTy = dyn_cast<cudaq::quake::VeqType>(type)) {
        if (!veqTy.hasSpecifiedSize())
          return std::nullopt;
        count += veqTy.getSize();
        continue;
      }
      if (cudaq::quake::isQuantumType(type))
        return std::nullopt;
    }
    return count;
  }

  bool isAllowedReferenceMappingScaffold(Operation *op) {
    return isa<cudaq::quake::AllocaOp, cudaq::quake::ConcatOp,
               cudaq::quake::DeallocOp, cudaq::quake::DiscriminateOp,
               cudaq::quake::ExtractRefOp, cudaq::quake::ReturnWireOp,
               cudaq::quake::SinkOp, cudaq::quake::UnwrapOp,
               cudaq::quake::WrapOp>(op);
  }

  void setIdentityMappingAttrs(func::FuncOp func, unsigned numQubits,
                               unsigned numMeasurements) {
    Builder builder(func.getContext());

    SmallVector<Attribute> v2p(numQubits);
    for (unsigned i = 0; i < numQubits; ++i)
      v2p[i] = builder.getI64IntegerAttr(i);
    func->setAttr("mapping_v2p", builder.getArrayAttr(v2p));

    SmallVector<Attribute> reorder(numMeasurements);
    for (unsigned i = 0; i < numMeasurements; ++i)
      reorder[i] = builder.getI64IntegerAttr(i);
    func->setAttr("mapping_reorder_idx", builder.getArrayAttr(reorder));
  }

  struct AffineIndex {
    Value base;
    int64_t offset = 0;
    Type originalType;
    Type extractedType;
  };

  struct ReferenceAccess {
    Value ref;
    Value veq;
    AffineIndex index;
  };

  struct IndexRange {
    int64_t min;
    int64_t max;
  };

  std::optional<int64_t> getConstantInt(Value value) {
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant)))
      return constant.getSExtValue();
    return std::nullopt;
  }

  std::optional<AffineIndex> getAffineIndex(Value value, Type extractedType) {
    if (auto constant = getConstantInt(value))
      return AffineIndex{Value{}, *constant, value.getType(), extractedType};

    if (auto cast = value.getDefiningOp<cudaq::cc::CastOp>())
      if (auto inner = getAffineIndex(cast.getValue(), value.getType())) {
        inner->extractedType = value.getType();
        return inner;
      }

    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
      if (auto rhs = getConstantInt(add.getRhs()))
        if (auto lhs = getAffineIndex(add.getLhs(), value.getType())) {
          lhs->offset += *rhs;
          lhs->originalType = value.getType();
          return lhs;
        }
      if (auto lhs = getConstantInt(add.getLhs()))
        if (auto rhs = getAffineIndex(add.getRhs(), value.getType())) {
          rhs->offset += *lhs;
          rhs->originalType = value.getType();
          return rhs;
        }
    }

    if (isa<IntegerType, IndexType>(value.getType()))
      return AffineIndex{value, 0, value.getType(), extractedType};
    return std::nullopt;
  }

  std::optional<ReferenceAccess>
  getReferenceAccess(Value ref, const DenseMap<Value, unsigned> &staticRefs) {
    if (auto iter = staticRefs.find(ref); iter != staticRefs.end())
      return ReferenceAccess{ref, Value{},
                             AffineIndex{Value{},
                                         static_cast<int64_t>(iter->second),
                                         Type{}, Type{}}};

    auto extract = ref.getDefiningOp<cudaq::quake::ExtractRefOp>();
    if (!extract)
      return std::nullopt;

    if (auto concat =
            extract.getVeq().getDefiningOp<cudaq::quake::ConcatOp>()) {
      for (auto target : concat.getTargets())
        if (!isa<cudaq::quake::RefType>(target.getType()))
          return std::nullopt;
    } else {
      auto veqType =
          dyn_cast<cudaq::quake::VeqType>(extract.getVeq().getType());
      if (!veqType || !veqType.hasSpecifiedSize())
        return std::nullopt;
    }

    AffineIndex index;
    if (extract.hasConstantIndex()) {
      index =
          AffineIndex{Value{}, static_cast<int64_t>(extract.getConstantIndex()),
                      Type{}, Type{}};
    } else {
      auto parsed =
          getAffineIndex(extract.getIndex(), extract.getIndex().getType());
      if (!parsed)
        return std::nullopt;
      index = *parsed;
    }
    return ReferenceAccess{ref, extract.getVeq(), index};
  }

  std::optional<int64_t> getConstantDistance(const ReferenceAccess &from,
                                             const ReferenceAccess &to) {
    if (from.veq != to.veq)
      return std::nullopt;
    if (from.index.base != to.index.base)
      return std::nullopt;
    return to.index.offset - from.index.offset;
  }

  std::optional<IndexRange> getLoopInductionRange(Value base) {
    auto arg = dyn_cast<BlockArgument>(base);
    if (!arg)
      return std::nullopt;

    auto loop =
        dyn_cast_or_null<cudaq::cc::LoopOp>(arg.getOwner()->getParentOp());
    if (!loop || arg.getOwner()->getParent() != &loop.getBodyRegion())
      return std::nullopt;

    cudaq::opt::LoopComponents components;
    if (!cudaq::opt::isaMonotonicLoop(loop.getOperation(),
                                      /*allowEarlyExit=*/false, &components))
      return std::nullopt;
    if (arg.getArgNumber() != components.induction)
      return std::nullopt;

    auto iterations = components.getIterationsConstant();
    auto initial = getConstantInt(components.initialValue);
    auto step = getConstantInt(components.stepValue);
    if (!iterations || *iterations == 0 || !initial || !step)
      return std::nullopt;
    if (!components.stepIsAnAddOp())
      *step = -*step;

    auto last = *initial + static_cast<int64_t>(*iterations - 1) * *step;
    return IndexRange{std::min(*initial, last), std::max(*initial, last)};
  }

  cudaq::cc::LoopOp getStaticLoopInductionOwner(Value base) {
    auto arg = dyn_cast<BlockArgument>(base);
    if (!arg)
      return {};

    auto loop =
        dyn_cast_or_null<cudaq::cc::LoopOp>(arg.getOwner()->getParentOp());
    if (!loop || arg.getOwner()->getParent() != &loop.getBodyRegion())
      return {};

    cudaq::opt::LoopComponents components;
    if (!cudaq::opt::isaMonotonicLoop(loop.getOperation(),
                                      /*allowEarlyExit=*/false, &components))
      return {};
    if (arg.getArgNumber() != components.induction)
      return {};
    if (!components.getIterationsConstant())
      return {};

    return loop;
  }

  std::optional<IndexRange>
  getReferenceIndexRange(const ReferenceAccess &access, unsigned numQubits) {
    IndexRange range;
    if (access.index.base) {
      auto baseRange = getLoopInductionRange(access.index.base);
      if (!baseRange)
        return std::nullopt;
      range = {baseRange->min + access.index.offset,
               baseRange->max + access.index.offset};
    } else {
      range = {access.index.offset, access.index.offset};
    }

    if (range.min < 0 || range.max >= static_cast<int64_t>(numQubits))
      return std::nullopt;
    return range;
  }

  bool isLinearRouteSupported(const cudaq::Device &device,
                              IndexRange routeRange) {
    if (routeRange.min < 0 ||
        routeRange.max >= static_cast<int64_t>(device.getNumQubits()))
      return false;

    for (auto index = routeRange.min; index < routeRange.max; ++index)
      if (!device.areConnected(
              cudaq::Device::Qubit(static_cast<unsigned>(index)),
              cudaq::Device::Qubit(static_cast<unsigned>(index + 1))))
        return false;
    return true;
  }

  std::optional<SmallVector<cudaq::Device::Qubit>>
  getShortestDevicePath(const cudaq::Device &device, cudaq::Device::Qubit src,
                        cudaq::Device::Qubit dst) {
    auto numQubits = device.getNumQubits();
    if (src.index >= numQubits || dst.index >= numQubits)
      return std::nullopt;

    SmallVector<int64_t> parent(numQubits, -1);
    SmallVector<cudaq::Device::Qubit> queue;
    queue.reserve(numQubits);
    parent[src.index] = src.index;
    queue.push_back(src);

    for (std::size_t begin = 0; begin < queue.size(); ++begin) {
      auto node = queue[begin];
      if (node == dst)
        break;
      for (auto neighbour : device.getNeighbours(node)) {
        if (neighbour.index >= numQubits || parent[neighbour.index] != -1)
          continue;
        parent[neighbour.index] = node.index;
        queue.push_back(neighbour);
      }
    }

    if (parent[dst.index] == -1)
      return std::nullopt;

    SmallVector<cudaq::Device::Qubit> path;
    for (auto node = dst.index;; node = parent[node]) {
      path.push_back(cudaq::Device::Qubit(static_cast<unsigned>(node)));
      if (node == src.index)
        break;
    }
    std::reverse(path.begin(), path.end());
    return path;
  }

  Value createIndexValue(OpBuilder &builder, Location loc,
                         const AffineIndex &baseIndex, int64_t offset) {
    if (!baseIndex.base)
      return Value{};

    Value index = baseIndex.base;
    if (offset != 0) {
      Value constant =
          isa<IndexType>(index.getType())
              ? Value(arith::ConstantIndexOp::create(builder, loc, offset))
              : Value(arith::ConstantIntOp::create(
                    builder, loc, cast<IntegerType>(index.getType()), offset));
      index = arith::AddIOp::create(builder, loc, index, constant);
    }

    if (baseIndex.extractedType && index.getType() != baseIndex.extractedType)
      index = cudaq::cc::CastOp::create(builder, loc, baseIndex.extractedType,
                                        index, cudaq::cc::CastOpMode::Signed);
    return index;
  }

  Value createReferenceAt(OpBuilder &builder, Location loc,
                          const ReferenceAccess &anchor, int64_t offset) {
    if (!anchor.veq)
      return Value{};

    if (!anchor.index.base)
      return cudaq::quake::ExtractRefOp::create(
          builder, loc, anchor.veq, static_cast<std::size_t>(offset));

    auto index = createIndexValue(builder, loc, anchor.index, offset);
    if (!index)
      return Value{};
    return cudaq::quake::ExtractRefOp::create(builder, loc, anchor.veq, index);
  }

  cudaq::quake::WrapOp findWrap(Value wire) {
    for (auto *user : wire.getUsers())
      if (auto wrap = dyn_cast<cudaq::quake::WrapOp>(user))
        return wrap;
    return {};
  }

  Value getStaticReferenceByIndex(const DenseMap<Value, unsigned> &staticRefs,
                                  unsigned index) {
    for (auto [ref, refIndex] : staticRefs)
      if (refIndex == index)
        return ref;
    return {};
  }

  Value createConstantReferenceAt(OpBuilder &builder, Location loc,
                                  const ReferenceAccess &lhsAccess,
                                  const ReferenceAccess &rhsAccess,
                                  const DenseMap<Value, unsigned> &staticRefs,
                                  unsigned index) {
    if (!lhsAccess.index.base && lhsAccess.index.offset >= 0 &&
        static_cast<unsigned>(lhsAccess.index.offset) == index)
      return lhsAccess.ref;
    if (!rhsAccess.index.base && rhsAccess.index.offset >= 0 &&
        static_cast<unsigned>(rhsAccess.index.offset) == index)
      return rhsAccess.ref;

    Value veq = lhsAccess.veq ? lhsAccess.veq : rhsAccess.veq;
    if (veq)
      return cudaq::quake::ExtractRefOp::create(
          builder, loc, veq, static_cast<std::size_t>(index));

    return getStaticReferenceByIndex(staticRefs, index);
  }

  LogicalResult insertReferenceRouteSwaps(Operation *op,
                                          ArrayRef<Value> routeRefs) {
    auto operands = cudaq::quake::getQuantumOperands(op);
    auto results = cudaq::quake::getQuantumResults(op);
    if (operands.size() != 2 || results.size() != 2)
      return failure();

    if (routeRefs.size() < 2)
      return failure();
    if (routeRefs.size() == 2)
      return success();

    auto routeDistance = routeRefs.size() - 1;
    SmallVector<Value> wires(routeRefs.size());
    wires[0] = operands[0];
    wires[routeDistance] = operands[1];

    auto loc = op->getLoc();
    OpBuilder beforeBuilder(op);
    for (unsigned i = 1; i < routeDistance; ++i)
      wires[i] = cudaq::quake::UnwrapOp::create(
          beforeBuilder, loc, operands[0].getType(), routeRefs[i]);

    auto wireType = operands[0].getType();
    for (unsigned i = routeDistance - 1; i > 0; --i) {
      auto swap = cudaq::quake::SwapOp::create(
          beforeBuilder, loc, TypeRange{wireType, wireType}, false,
          ValueRange{}, ValueRange{}, ValueRange{wires[i], wires[i + 1]},
          DenseBoolArrayAttr{});
      wires[i] = swap.getResult(0);
      wires[i + 1] = swap.getResult(1);
    }

    if (failed(cudaq::quake::setQuantumOperands(op, {wires[0], wires[1]})))
      return failure();
    wires[0] = results[0];
    wires[1] = results[1];

    OpBuilder afterBuilder(op->getContext());
    afterBuilder.setInsertionPointAfter(op);
    for (unsigned i = 1; i < routeDistance; ++i) {
      auto swap = cudaq::quake::SwapOp::create(
          afterBuilder, loc, TypeRange{wireType, wireType}, false, ValueRange{},
          ValueRange{}, ValueRange{wires[i], wires[i + 1]},
          DenseBoolArrayAttr{});
      wires[i] = swap.getResult(0);
      wires[i + 1] = swap.getResult(1);
    }

    auto lhsWrap = findWrap(results[0]);
    auto rhsWrap = findWrap(results[1]);
    if (!lhsWrap || !rhsWrap)
      return failure();
    lhsWrap.getWireValueMutable().assign(wires[0]);
    rhsWrap.getWireValueMutable().assign(wires[routeDistance]);
    for (unsigned i = 1; i < routeDistance; ++i)
      cudaq::quake::WrapOp::create(afterBuilder, loc, wires[i], routeRefs[i]);

    return success();
  }

  LogicalResult
  routeStaticReferenceQlsTwoQubitOp(Operation *op, const cudaq::Device &device,
                                    const DenseMap<Value, unsigned> &staticRefs,
                                    const ReferenceAccess &lhsAccess,
                                    const ReferenceAccess &rhsAccess,
                                    unsigned numQubits) {
    if (lhsAccess.index.base || rhsAccess.index.base)
      return failure();
    if (lhsAccess.index.offset < 0 || rhsAccess.index.offset < 0)
      return failure();

    auto lhsIndex = static_cast<unsigned>(lhsAccess.index.offset);
    auto rhsIndex = static_cast<unsigned>(rhsAccess.index.offset);
    if (lhsIndex >= numQubits || rhsIndex >= numQubits ||
        lhsIndex >= device.getNumQubits() || rhsIndex >= device.getNumQubits())
      return failure();

    auto path = getShortestDevicePath(device, cudaq::Device::Qubit(lhsIndex),
                                      cudaq::Device::Qubit(rhsIndex));
    if (!path || path->size() < 2)
      return failure();

    OpBuilder builder(op);
    SmallVector<Value> routeRefs;
    routeRefs.reserve(path->size());
    for (auto qubit : *path) {
      auto ref = createConstantReferenceAt(builder, op->getLoc(), lhsAccess,
                                           rhsAccess, staticRefs, qubit.index);
      if (!ref)
        return failure();
      routeRefs.push_back(ref);
    }

    return insertReferenceRouteSwaps(op, routeRefs);
  }

  LogicalResult
  routeReferenceQlsTwoQubitOp(Operation *op, const cudaq::Device &device,
                              const DenseMap<Value, unsigned> &staticRefs,
                              unsigned numQubits) {
    auto operands = cudaq::quake::getQuantumOperands(op);
    auto results = cudaq::quake::getQuantumResults(op);
    if (operands.size() != 2 || results.size() != 2)
      return failure();

    auto lhsUnwrap = operands[0].getDefiningOp<cudaq::quake::UnwrapOp>();
    auto rhsUnwrap = operands[1].getDefiningOp<cudaq::quake::UnwrapOp>();
    if (!lhsUnwrap || !rhsUnwrap)
      return failure();

    auto lhsAccess = getReferenceAccess(lhsUnwrap.getRefValue(), staticRefs);
    auto rhsAccess = getReferenceAccess(rhsUnwrap.getRefValue(), staticRefs);
    if (!lhsAccess || !rhsAccess)
      return failure();

    auto distance = getConstantDistance(*lhsAccess, *rhsAccess);
    if (!distance)
      return failure();
    if (*distance == 0)
      return failure();
    auto absDistance = static_cast<unsigned>(std::llabs(*distance));

    if (!lhsAccess->index.base)
      return routeStaticReferenceQlsTwoQubitOp(
          op, device, staticRefs, *lhsAccess, *rhsAccess, numQubits);

    auto lhsRange = getReferenceIndexRange(*lhsAccess, numQubits);
    auto rhsRange = getReferenceIndexRange(*rhsAccess, numQubits);
    if (!lhsRange || !rhsRange)
      return failure();

    IndexRange routeRange{std::min(lhsRange->min, rhsRange->min),
                          std::max(lhsRange->max, rhsRange->max)};
    if (!isLinearRouteSupported(device, routeRange))
      return failure();
    if (absDistance == 1)
      return success();

    SmallVector<Value> refs(absDistance + 1);
    refs[0] = lhsAccess->ref;
    refs[absDistance] = rhsAccess->ref;

    auto loc = op->getLoc();
    OpBuilder beforeBuilder(op);
    auto sign = *distance > 0 ? 1 : -1;
    for (unsigned i = 1; i < absDistance; ++i) {
      refs[i] = createReferenceAt(beforeBuilder, loc, *lhsAccess,
                                  lhsAccess->index.offset + sign * i);
      if (!refs[i])
        return failure();
    }

    return insertReferenceRouteSwaps(op, refs);
  }

  void collectStaticLoopInductionDependencies(
      Value value, llvm::SmallPtrSetImpl<Operation *> &loops,
      llvm::SmallPtrSetImpl<Value> &visited) {
    if (!value || !visited.insert(value).second)
      return;

    if (auto loop = getStaticLoopInductionOwner(value)) {
      loops.insert(loop.getOperation());
      return;
    }

    auto *def = value.getDefiningOp();
    if (!def)
      return;

    for (auto operand : def->getOperands())
      collectStaticLoopInductionDependencies(operand, loops, visited);
  }

  void collectReferenceIndexLoops(Value ref,
                                  llvm::SmallPtrSetImpl<Operation *> &loops) {
    auto extract = ref.getDefiningOp<cudaq::quake::ExtractRefOp>();
    if (!extract || extract.hasConstantIndex())
      return;

    llvm::SmallPtrSet<Value, 16> visited;
    collectStaticLoopInductionDependencies(extract.getIndex(), loops, visited);
  }

  bool selectivelyUnrollReferenceIndexLoops(func::FuncOp func) {
    llvm::SmallPtrSet<Operation *, 8> loopsToUnroll;

    func.walk([&](cudaq::quake::OperatorInterface gate) {
      auto controlCount = getSingleReferenceOperandCount(gate.getControls());
      auto targetCount = getSingleReferenceOperandCount(gate.getTargets());
      if (!controlCount || !targetCount || *controlCount + *targetCount != 2)
        return WalkResult::advance();

      auto operands = cudaq::quake::getQuantumOperands(gate);
      if (operands.size() != 2)
        return WalkResult::advance();

      auto lhsUnwrap = operands[0].getDefiningOp<cudaq::quake::UnwrapOp>();
      auto rhsUnwrap = operands[1].getDefiningOp<cudaq::quake::UnwrapOp>();
      if (!lhsUnwrap || !rhsUnwrap)
        return WalkResult::advance();

      collectReferenceIndexLoops(lhsUnwrap.getRefValue(), loopsToUnroll);
      collectReferenceIndexLoops(rhsUnwrap.getRefValue(), loopsToUnroll);
      return WalkResult::advance();
    });

    if (loopsToUnroll.empty())
      return false;

    auto *ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    for (auto *dialect : ctx->getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : ctx->getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, ctx);

    unsigned progress = 0;
    patterns.insert<UnrollCountedLoop>(
        ctx, /*threshold=*/1024, /*signalFailure=*/false,
        /*allowBreak=*/false, progress,
        [loopsToUnroll](cudaq::cc::LoopOp loop) {
          return loopsToUnroll.contains(loop.getOperation());
        });

    FrozenRewritePatternSet frozen(std::move(patterns));
    bool changed = false;
    do {
      progress = 0;
      (void)applyPatternsGreedily(func, frozen);
      changed |= progress != 0;
    } while (progress);

    return changed;
  }

  LogicalResult tryReferenceQlsMappingInPlace(func::FuncOp func,
                                              const cudaq::Device &device) {
    auto numQubits = getStaticReferenceQubitCount(func);
    if (!numQubits)
      return failure();
    if (*numQubits == 0 || *numQubits > device.getNumQubits())
      return failure();

    DenseMap<Value, unsigned> staticRefs;
    unsigned refIndex = 0;
    func.walk([&](cudaq::quake::AllocaOp alloca) {
      staticRefs[alloca.getResult()] = refIndex++;
    });

    bool hasQuantum = false;
    unsigned numMeasurements = 0;
    auto walkResult = func.walk([&](Operation *op) {
      if (isa<cudaq::quake::NullWireOp>(op))
        return WalkResult::interrupt();

      if (auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op)) {
        hasQuantum = true;
        auto controlCount = getSingleReferenceOperandCount(gate.getControls());
        auto targetCount = getSingleReferenceOperandCount(gate.getTargets());
        if (!controlCount || !targetCount)
          return WalkResult::interrupt();
        auto totalCount = *controlCount + *targetCount;
        if (totalCount > 2)
          return WalkResult::interrupt();
        if (totalCount == 2 && failed(routeReferenceQlsTwoQubitOp(
                                   op, device, staticRefs, *numQubits)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }

      if (auto meas = dyn_cast<cudaq::quake::MeasurementInterface>(op)) {
        hasQuantum = true;
        auto count = getMeasurementOperandCount(meas.getTargets());
        if (!count)
          return WalkResult::interrupt();
        numMeasurements += *count;
        return WalkResult::advance();
      }

      if (isAllowedReferenceMappingScaffold(op)) {
        hasQuantum = true;
        return WalkResult::advance();
      }

      if (op->getName().getStringRef().starts_with("quake."))
        return WalkResult::interrupt();

      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      return failure();
    if (!hasQuantum)
      return failure();
    setIdentityMappingAttrs(func, *numQubits, numMeasurements);
    return success();
  }

  LogicalResult tryReferenceQlsMapping(func::FuncOp func,
                                       const cudaq::Device &device,
                                       bool allowSelectiveUnroll = true) {
    if (!allowSelectiveUnroll)
      return tryReferenceQlsMappingInPlace(func, device);

    auto trial = cast<func::FuncOp>(func->clone());
    auto trialResult = tryReferenceQlsMappingInPlace(trial, device);
    trial->destroy();
    if (succeeded(trialResult))
      return tryReferenceQlsMappingInPlace(func, device);

    if (!selectivelyUnrollReferenceIndexLoops(func))
      return failure();

    if (succeeded(tryReferenceQlsMapping(func, device,
                                         /*allowSelectiveUnroll=*/false)))
      return success();
    return failure();
  }

  void runOnOperation() override {
    if (deviceBypass)
      return;

    auto func = getOperation();
    if (func.empty())
      return;
    auto &blocks = func.getBlocks();

    // Current limitations:
    //  * Can only map a entry-point kernel
    //  * The kernel can only have one block

    auto mod = func->getParentOfType<ModuleOp>();
    auto wireSetOp =
        mod.lookupSymbol<cudaq::quake::WireSetOp>(mappedWireSetName);
    if (!wireSetOp) {
      // Silently return without error if no mapped wire set is found in the
      // module.
      return;
    }

    // FIXME: Add the ability to handle multiple blocks.
    if (blocks.size() > 1) {
      if (nonComposable) {
        func.emitError("The mapper cannot handle multiple blocks");
        signalPassFailure();
      }
      LLVM_DEBUG(llvm::dbgs() << "NYI: mapping with multiple blocks");
      return;
    }

    // Verify that the function contains wiresets and return if it does not.
    // Also populate the highest identity borrow up as long as we're traversing
    // them.
    StringRef inputWireSet;
    std::optional<std::uint32_t> highestIdentity;
    auto walkResult = func.walk([&](cudaq::quake::BorrowWireOp borrowOp) {
      if (inputWireSet.empty()) {
        inputWireSet = borrowOp.getSetName();
      } else if (borrowOp.getSetName() != inputWireSet) {
        // Why is this here? It's entirely possible to have disjoint wire sets,
        // where the sets are for fundamentally distinct purposes in the target
        // model.
        if (nonComposable)
          func.emitOpError("function cannot use multiple WireSets");
        return WalkResult::interrupt();
      }
      highestIdentity = highestIdentity
                            ? std::max(*highestIdentity, borrowOp.getIdentity())
                            : borrowOp.getIdentity();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      if (nonComposable)
        signalPassFailure();
      LLVM_DEBUG(llvm::dbgs()
                 << "NYI: multiple wire sets for a target machine");
      return;
    }
    if (!highestIdentity) {
      if (deviceInstance->getNumQubits() != 0 &&
          succeeded(tryReferenceQlsMapping(func, *deviceInstance)))
        return;

      if (nonComposable) {
        func.emitOpError("no borrow_wire ops found in " + func.getName());
        signalPassFailure();
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "no borrow_wire ops found in " << func.getName() << '\n');
      return;
    }

    // Sanity checks and create a wire to virtual qubit mapping.
    Block &block = *blocks.begin();

    if (deviceInstance->getNumQubits() == 0) {
      if (nonComposable) {
        func.emitError("Trying to target an empty device.");
        signalPassFailure();
      }
      LLVM_DEBUG(llvm::dbgs() << "device cannot be empty");
      return;
    }

    LLVM_DEBUG({ deviceInstance->dump(); });

    const std::size_t deviceNumQubits = deviceInstance->getNumQubits();

    SmallVector<cudaq::quake::BorrowWireOp> sources(deviceNumQubits);
    SmallVector<cudaq::quake::ReturnWireOp> returnsToRemove;
    DenseMap<Value, cudaq::Placement::VirtualQ> wireToVirtualQ;
    SmallVector<std::size_t> userQubitsMeasured;
    DenseMap<std::size_t, Value> finalQubitWire;
    Operation *lastSource = nullptr;
    for (Operation &op : block.getOperations()) {
      if (auto qop = dyn_cast<cudaq::quake::BorrowWireOp>(op)) {
        // Assign a new virtual qubit to the resulting wire.
        auto id = qop.getIdentity();
        wireToVirtualQ[qop.getResult()] = cudaq::Placement::VirtualQ(id);
        finalQubitWire[id] = qop.getResult();
        sources[id] = qop;
        lastSource = &op;
      } else if (dyn_cast<cudaq::quake::NullWireOp>(op)) {
        if (nonComposable) {
          op.emitOpError(
              "the mapper requires borrow operations and prohibits null wires");
          signalPassFailure();
        }
        LLVM_DEBUG(llvm::dbgs() << "null_wire ops are not expected");
        return;
      } else if (dyn_cast<cudaq::quake::AllocaOp>(op)) {
        if (nonComposable) {
          op.emitOpError("the mapper requires borrow operations and prohibits "
                         "reference semantics");
          signalPassFailure();
        }
        LLVM_DEBUG(llvm::dbgs() << "quantum reference semantics not expected");
        return;
      } else if (cudaq::quake::isSupportedMappingOperation(&op)) {
        // Make sure the operation is using value semantics.
        if (!cudaq::quake::isLinearValueForm(&op)) {
          if (nonComposable) {
            llvm::errs() << "This is not SSA form: " << op << '\n';
            llvm::errs() << "isa<cudaq::quake::NullWireOp>() = "
                         << isa<cudaq::quake::NullWireOp>(&op) << '\n';
            llvm::errs() << "isAllReferences() = "
                         << cudaq::quake::isAllReferences(&op) << '\n';
            llvm::errs() << "isWrapped() = " << cudaq::quake::isWrapped(&op)
                         << '\n';
            func.emitError("The mapper requires value semantics.");
            signalPassFailure();
          }
          LLVM_DEBUG(llvm::dbgs() << "operation is not in proper value form");
          return;
        }

        // Since `quake.return_wire` operations do not generate new wires, we
        // don't need to further analyze.
        if (auto rop = dyn_cast<cudaq::quake::ReturnWireOp>(op)) {
          returnsToRemove.push_back(rop);
          continue;
        }

        // Get the wire operands and check if the operators uses at most two
        // qubits. N.B: Measurements do not have this restriction.
        auto wireOperands = cudaq::quake::getQuantumOperands(&op);
        if (!op.hasTrait<cudaq::QuantumMeasure>() && wireOperands.size() > 2) {
          if (nonComposable) {
            func.emitError("Cannot map a kernel with operators that use more "
                           "than two qubits.");
            signalPassFailure();
          }
          LLVM_DEBUG(llvm::dbgs() << "operator with >2 qubits not expected");
          return;
        }

        // Save which qubits are measured
        if (isa<cudaq::quake::MeasurementInterface>(op))
          for (const auto &wire : wireOperands)
            userQubitsMeasured.push_back(wireToVirtualQ[wire].index);

        // Map the result wires to the appropriate virtual qubits.
        for (auto &&[wire, newWire] : llvm::zip_equal(
                 wireOperands, cudaq::quake::getQuantumResults(&op))) {
          // Don't use wireToVirtualQ[a] = wireToVirtualQ[b]. It will work
          // *most* of the time but cause memory corruption other times because
          // DenseMap references can be invalidated upon insertion of new pairs.
          wireToVirtualQ.insert({newWire, wireToVirtualQ[wire]});
          finalQubitWire[wireToVirtualQ[wire].index] = newWire;
        }
      }
    }

    if (sources.size() > deviceNumQubits) {
      if (nonComposable) {
        func.emitOpError("Too many qubits [" + std::to_string(sources.size()) +
                         "] for device [" + std::to_string(deviceNumQubits) +
                         "]");
        signalPassFailure();
      }
      LLVM_DEBUG(llvm::dbgs() << "exceeded available qubits for target");
      return;
    }

    // Make all existing borrow_wire ops use the mapped wire set.
    func.walk([&](cudaq::quake::BorrowWireOp borrowOp) {
      borrowOp.setSetName(mappedWireSetName);
    });

    // We've made it past all the initial checks. Remove the returns now. They
    // will be added back in when the mapping is complete.
    for (auto ret : returnsToRemove)
      ret.erase();
    returnsToRemove.clear();

    OpBuilder builder(&block, block.begin());
    auto wireTy = builder.getType<cudaq::quake::WireType>();
    auto unknownLoc = builder.getUnknownLoc();

    // Add implicit measurements if necessary
    if (userQubitsMeasured.empty()) {
      builder.setInsertionPoint(block.getTerminator());
      auto measTy = cudaq::quake::MeasureType::get(builder.getContext());
      Type resTy = builder.getI1Type();
      for (unsigned i = 0; i < sources.size(); i++) {
        if (sources[i] != nullptr) {
          auto measureOp = cudaq::quake::MzOp::create(
              builder, finalQubitWire[i].getLoc(), TypeRange{measTy, wireTy},
              finalQubitWire[i]);
          cudaq::quake::DiscriminateOp::create(builder,
                                               finalQubitWire[i].getLoc(),
                                               resTy, measureOp.getMeasOut());

          wireToVirtualQ.insert(
              {measureOp.getWires()[0], wireToVirtualQ[finalQubitWire[i]]});

          userQubitsMeasured.push_back(i);
        }
      }
    }

    // Save the order of the measurements. They are not allowed to change.
    SmallVector<mlir::Operation *> measureOrder;
    func.walk([&](cudaq::quake::MeasurementInterface measure) {
      measureOrder.push_back(measure);
      for (auto user : measure->getUsers())
        measureOrder.push_back(user);
      return WalkResult::advance();
    });

    // Create or borrow auxillary qubits if needed. Place them after the last
    // allocated qubit.
    builder.setInsertionPointAfter(lastSource);
    for (unsigned i = 0; i < deviceInstance->getNumQubits(); i++) {
      if (!sources[i]) {
        auto borrowOp = cudaq::quake::BorrowWireOp::create(
            builder, unknownLoc, wireTy, mappedWireSetName, i);
        wireToVirtualQ[borrowOp.getResult()] = cudaq::Placement::VirtualQ(i);
        sources[i] = borrowOp;
      }
    }

    // Place
    cudaq::Placement placement(sources.size(), deviceInstance->getNumQubits());
    identityPlacement(placement);

    // Route
    SabreRouter router(*deviceInstance, wireToVirtualQ, placement,
                       extendedLayerSize, extendedLayerWeight, decayDelta,
                       roundsDecayReset);
    router.route(*blocks.begin(), sources);
    sortTopologically(&block);

    // Ensure that the original measurement ordering is still honored by moving
    // the measurements to the end (in their original order). Note that we must
    // move the users of those measurements to the end as well.
    for (Operation *measure : measureOrder) {
      SmallVector<Operation *> opsToMoveToEnd;
      addOpAndUsersToList(measure, opsToMoveToEnd);
      for (Operation *op : opsToMoveToEnd)
        block.getOperations().splice(std::prev(block.end()),
                                     block.getOperations(), op->getIterator());
    }

    // Remove any unused BorrowWireOps and add ReturnWireOp's where needed
    // unsigned highestMappedQubit = 0;
    builder.setInsertionPoint(block.getTerminator());
    auto phyToWire = router.getPhyToWire();
    for (const auto &[i, s] : llvm::enumerate(sources)) {
      if (s->getUsers().empty()) {
        s->erase();
      } else {
        // highestMappedQubit = i;
        cudaq::quake::ReturnWireOp::create(builder, phyToWire[i].getLoc(),
                                           phyToWire[i]);
      }
    }

    // Populate mapping_v2p attribute on this function such that:
    // - mapping_v2p[v] contains the final physical qubit placement for virtual
    //   qubit `v`.
    // To map the backend qubits back to the original user program (i.e. before
    // this pass), run something like this:
    //   for (int v = 0; v < numQubits; v++)
    //     dataForOriginalQubit[v] = dataFromBackendQubit[mapping_v2p[v]];
    llvm::SmallVector<Attribute> attrs(*highestIdentity + 1);
    for (unsigned int v = 0; v < *highestIdentity + 1; v++)
      attrs[v] = IntegerAttr::get(
          builder.getIntegerType(64),
          placement.getPhy(cudaq::Placement::VirtualQ(v)).index);

    func->setAttr("mapping_v2p", builder.getArrayAttr(attrs));

    // Now populate mapping_reorder_idx attribute. This attribute will be used
    // by downstream processing to reconstruct a global register as if mapping
    // had not occurred. This is important because the global register is
    // required to be sorted by qubit allocation order, and mapping can change
    // that apparent order AND introduce ancilla qubits that we don't want to
    // appear in the final global register.

    // pair is <first=virtual, second=physical>
    using VirtPhyPairType = std::pair<std::size_t, std::size_t>;
    llvm::SmallVector<VirtPhyPairType> measuredQubits;
    measuredQubits.reserve(userQubitsMeasured.size());
    for (auto mq : userQubitsMeasured) {
      measuredQubits.emplace_back(
          mq, placement.getPhy(cudaq::Placement::VirtualQ(mq)).index);
    }
    // First sort the pairs according to the physical qubits.
    llvm::sort(measuredQubits,
               [&](const VirtPhyPairType &a, const VirtPhyPairType &b) {
                 return a.second < b.second;
               });
    // Now find out how to reorder `measuredQubits` such that the elements are
    // ordered based on the *virtual* qubits (i.e. measuredQubits[].first).
    llvm::SmallVector<std::size_t> reorder_idx(measuredQubits.size());
    for (std::size_t ix = 0; auto &element : reorder_idx)
      element = ix++;
    llvm::sort(reorder_idx, [&](const std::size_t &i1, const std::size_t &i2) {
      return measuredQubits[i1].first < measuredQubits[i2].first;
    });
    // After kernel execution is complete, you can pass reorder_idx[] into
    // sample_result::reorder() in order to undo the ordering change to the
    // global register that the mapping pass induced.
    llvm::SmallVector<Attribute> mapping_reorder_idx(reorder_idx.size());
    for (std::size_t ix = 0; auto &element : mapping_reorder_idx)
      element = IntegerAttr::get(builder.getIntegerType(64), reorder_idx[ix++]);

    func->setAttr("mapping_reorder_idx",
                  builder.getArrayAttr(mapping_reorder_idx));
  }
};

} // namespace

namespace cudaq::opt {
/// This options structure is mostly a mirror copy of the options in
/// MappingFunc, but we've also added the `device` option from MappingPrep.
struct MappingPipelineOptions
    : public PassPipelineOptions<MappingPipelineOptions> {

#define DECLARE_SUB_OPTION(_PARENT_STRUCT, _FIELD)                             \
  PassOptions::Option<decltype(_PARENT_STRUCT::_FIELD)> _FIELD{*this, #_FIELD}
  DECLARE_SUB_OPTION(MappingPrepOptions, device);
  DECLARE_SUB_OPTION(MappingFuncOptions, extendedLayerSize);
  DECLARE_SUB_OPTION(MappingFuncOptions, extendedLayerWeight);
  DECLARE_SUB_OPTION(MappingFuncOptions, decayDelta);
  DECLARE_SUB_OPTION(MappingFuncOptions, roundsDecayReset);
  PassOptions::Option<bool> nonComposable{*this, "raise-fatal-errors"};
};

/// Register the mapping pipeline. Route the appropriate options to the
/// appropriate pass in the pass pipeline.
void registerMappingPipeline() {
  PassPipelineRegistration<cudaq::opt::MappingPipelineOptions>(
      "qubit-mapping", "Perform qubit mapping pass pipeline.",
      [](OpPassManager &pm, const MappingPipelineOptions &opt) {
        auto setIt = [](auto &to, const auto &from) {
          if (from.hasValue())
            to = from;
        };

        // Add the prep pass
        MappingPrepOptions prepOpts;
        setIt(prepOpts.device, opt.device);
        setIt(prepOpts.nonComposable, opt.nonComposable);
        pm.addPass(cudaq::opt::createMappingPrep(prepOpts));

        // Add the per-function pass
        MappingFuncOptions funcOpts;
        setIt(funcOpts.device, opt.device);
        setIt(funcOpts.extendedLayerSize, opt.extendedLayerSize);
        setIt(funcOpts.extendedLayerWeight, opt.extendedLayerWeight);
        setIt(funcOpts.decayDelta, opt.decayDelta);
        setIt(funcOpts.roundsDecayReset, opt.roundsDecayReset);
        setIt(funcOpts.nonComposable, opt.nonComposable);
        pm.addNestedPass<func::FuncOp>(cudaq::opt::createMappingFunc(funcOpts));
      });
}
} // namespace cudaq::opt
