//************************************************************************
//   proxy application v.0.0.1
//
//  semproxy.cpp: the main interface of  proxy application
//
//************************************************************************

#include "sem_proxy.h"
#include <rle.h>
#include "pressure_statistics.h"

#include <cartesian_struct_builder.h>
#include <cartesian_unstruct_builder.h>
#include <sem_solver_acoustic.h>
#include <source_and_receiver_utils.h>

#include <cxxopts.hpp>
#include <iostream>
#include <sstream>
#include <variant>
#include <cmath>
#include <cstdint>

#include <filesystem>
#include <fstream>
#include <iomanip>

using namespace SourceAndReceiverUtils;

// Shared field names for snapshot and receiver results, matching analyze.py.
static void writePressureStatistics(std::ostream& output,
                                   const PressureExtrema& extrema,
                                   const PressureMeanVariance& moments)
{
  output << "        \"minimum_pressure\": " << extrema.minimum_pressure << ",\n"
         << "        \"maximum_pressure\": " << extrema.maximum_pressure << ",\n"
         << "        \"maximum_absolute_pressure\": "
         << extrema.maximum_absolute_pressure << ",\n"
         << "        \"mean_pressure\": " << moments.mean_pressure << ",\n"
         << "        \"pressure_variance\": " << moments.pressure_variance << "\n";
}

SEMproxy::SEMproxy(const SemProxyOptions& opt)
{
  is_in_situ_ = opt.mode == "in-situ";
  extrema_backend_ = opt.extrema_backend;
  process_snapshots_ = opt.outputs != "sismograms";
  process_sismograms_ = opt.outputs != "snapshots";
  is_snapshots_ = opt.is_snapshots_ && !is_in_situ_ && process_snapshots_;
  snap_time_interval_ = opt.snap_time_interval_;
  snap_folder_ = opt.snap_folder_;

  if (is_snapshots_)
    std::filesystem::create_directories(snap_folder_);
  
  rle_ = opt.rle;
  rle_folder_ = opt.rle_folder;
  if (rle_)
  {
    std::filesystem::create_directories(rle_folder_);
    if (process_snapshots_)
      std::ofstream(std::filesystem::path(rle_folder_) / "rle_snapshots.csv")
          << "timestep,time_s,values,runs,raw_bytes,rle_bytes,ratio,"
             "copy_s,encode_s,write_s\n";
  }

  const int order = opt.order;
  nb_elements_[0] = opt.ex;
  nb_elements_[1] = opt.ey;
  nb_elements_[2] = opt.ez;
  nb_nodes_[0] = opt.ex * order + 1;
  nb_nodes_[1] = opt.ey * order + 1;
  nb_nodes_[2] = opt.ez * order + 1;

  const float spongex = opt.boundaries_size;
  const float spongey = opt.boundaries_size;
  const float spongez = opt.boundaries_size;
  const std::array<float, 3> sponge_size = {spongex, spongey, spongez};
  src_coord_[0] = opt.srcx;
  src_coord_[1] = opt.srcy;
  src_coord_[2] = opt.srcz;

  domain_size_[0] = opt.lx;
  domain_size_[1] = opt.ly;
  domain_size_[2] = opt.lz;

  rcv_coord_[0] = opt.rcvx;
  rcv_coord_[1] = opt.rcvy;
  rcv_coord_[2] = opt.rcvz;

  if (!process_sismograms_)
    {
      // An isolated snapshot workflow has no receiver interpolation or history.
      receiver_coords_.clear();
    }
  else if (opt.recepteur_file_.empty())
    {
      receiver_coords_.push_back(rcv_coord_);
    }
  else
    {
      std::ifstream input(opt.recepteur_file_);
      if (!input)
        throw std::runtime_error("Cannot open receiver file: " + opt.recepteur_file_);

      std::string line;
      while (std::getline(input, line))
        {
          if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
          std::istringstream values(line);
          std::array<float, 3> coords;
          std::string extra;
          if (!(values >> coords[0] >> coords[1] >> coords[2]) || values >> extra)
            throw std::runtime_error("Invalid receiver position: " + line);
          receiver_coords_.push_back(coords);
        }
      if (input.bad())
        throw std::runtime_error("Cannot read receiver file: " + opt.recepteur_file_);
      if (receiver_coords_.empty())
        throw std::runtime_error("Receiver file is empty: " + opt.recepteur_file_);
    }

  for (const auto& coords : receiver_coords_)
    for (int axis = 0; axis < 3; ++axis)
      if (!std::isfinite(coords[axis]) || coords[axis] < 0 ||
          coords[axis] >= domain_size_[axis])
        throw std::runtime_error("Receiver position outside domain");
  
  bool isModelOnNodes = opt.isModelOnNodes;
  isElastic_ = opt.isElastic;
  cout << boolalpha;
  bool isElastic = isElastic_;

  const SolverFactory::methodType methodType = getMethod(opt.method);
  const SolverFactory::implemType implemType = getImplem(opt.implem);
  const SolverFactory::meshType meshType = getMesh(opt.mesh);
  const SolverFactory::modelLocationType modelLocation =
      isModelOnNodes ? SolverFactory::modelLocationType::OnNodes
                     : SolverFactory::modelLocationType::OnElements;
  const SolverFactory::physicType physicType = SolverFactory::physicType::Acoustic;

  float lx = domain_size_[0];
  float ly = domain_size_[1];
  float lz = domain_size_[2];
  int ex = nb_elements_[0];
  int ey = nb_elements_[1];
  int ez = nb_elements_[2];

  if (meshType == SolverFactory::Struct)
  {
    switch (order)
    {
      case 1: {
        model::CartesianStructBuilder<float, int, 1> builder(
            ex, lx, ey, ly, ez, lz, isModelOnNodes);
        m_mesh = builder.getModel();
        break;
      }
      case 2: {
        model::CartesianStructBuilder<float, int, 2> builder(
            ex, lx, ey, ly, ez, lz, isModelOnNodes);
        m_mesh = builder.getModel();
        break;
      }
      case 3: {
        model::CartesianStructBuilder<float, int, 3> builder(
            ex, lx, ey, ly, ez, lz, isModelOnNodes);
        m_mesh = builder.getModel();
        break;
      }
      default:
        throw std::runtime_error(
            "Order other than 1 2 3 is not supported (semproxy)");
    }
  }
  else if (meshType == SolverFactory::Unstruct)
  {
    model::CartesianParams<float, int> param(order, ex, ey, ez, lx, ly, lz,
                                             isModelOnNodes);
    model::CartesianUnstructBuilder<float, int> builder(param);
    m_mesh = builder.getModel();
  }
  else
  {
    throw std::runtime_error("Incorrect mesh type (SEMproxy ctor.)");
  }

  // time parameters
  if (opt.autodt)
  {
    float cfl_factor = (order == 2) ? 0.5 : 0.7;
    dt_ = find_cfl_dt(cfl_factor);
  }
  else
  {
    dt_ = opt.dt;
  }
  timemax_ = opt.timemax;
  if (is_in_situ_ && (!std::isfinite(dt_) || dt_ <= 0 ||
                     !std::isfinite(timemax_) || timemax_ < dt_))
    throw std::runtime_error("In-situ statistics require a finite positive dt and timemax >= dt");
  num_sample_ = timemax_ / dt_;

  m_solver = SolverFactory::createSolver(methodType, implemType, meshType,
                                         modelLocation, physicType, order);
  m_solver->computeFEInit(*m_mesh, sponge_size, opt.surface_sponge,
                          opt.taper_delta);

  initFiniteElem();

  std::cout << "Number of node is " << m_mesh->getNumberOfNodes() << std::endl;
  std::cout << "Number of element is " << m_mesh->getNumberOfElements()
            << std::endl;
  std::cout << "Launching the Method " << opt.method << ", the implementation "
            << opt.implem << " and the mesh is " << opt.mesh << std::endl;
  std::cout << "Number of receivers is " << receiver_coords_.size() << std::endl;
  std::cout << "Model is on " << (isModelOnNodes ? "nodes" : "elements")
            << std::endl;
  std::cout << "Physics type is " << (isElastic ? "elastic" : "acoustic")
            << std::endl;
  std::cout << "Order of approximation will be " << order << std::endl;
  std::cout << "Time step is " << dt_ << "s" << std::endl;
  std::cout << "Simulated time is " << timemax_ << "s" << std::endl;

}

void SEMproxy::run()
{
  using Clock = std::chrono::steady_clock;
  metrics_ = {};
  mesh_saved_ = false;
  metrics_.in_situ = is_in_situ_;
  metrics_.outputs = process_snapshots_ ? (process_sismograms_ ? "both" : "snapshots")
                                       : "sismograms";
  metrics_.sample_count = num_sample_;
  metrics_.extrema_backend = extrema_backend_;
  metrics_.rle = rle_;
#ifdef USE_KOKKOS
  metrics_.execution_space = Kokkos::DefaultExecutionSpace::name();
  std::unique_ptr<PressureExtremaWorkspace> extrema_workspace;
  if (!extrema_backend_.empty())
    extrema_workspace = std::make_unique<PressureExtremaWorkspace>(
        m_mesh->getNumberOfNodes(), extrema_backend_ == "cpu");
#endif

  SEMsolverDataAcoustic solverData(i1, i2, myRHSTerm, pnGlobal, rhsElement,
                                   rhsWeights);

  std::ofstream statistics;
  const char* statistics_path = "statistics_in-situ.json";
  bool first_snapshot = true;
  PressureExtrema global_extrema{};
  std::vector<PressureExtrema> receiver_extrema;
  std::vector<WelfordState> receiver_moments;
  std::vector<float> receiver_pressures;
  if (is_in_situ_)
  {
    // O(receivers) storage, independent of the number of simulated time steps.
    receiver_extrema.resize(receiver_coords_.size());
    receiver_moments.resize(receiver_coords_.size());
    receiver_pressures.resize(receiver_coords_.size());
    const auto startWriteTime = Clock::now();
    statistics.open(statistics_path);
    if (!statistics) throw std::runtime_error("Cannot open in-situ statistics file");
    statistics << std::setprecision(17) << "{\n  \"snapshots\": ";
    if (process_snapshots_)
      statistics << "{\n    \"per_snapshot\": [";
    else
      statistics << "null";
    if (!statistics) throw std::runtime_error("Cannot write in-situ statistics file");
    metrics_.statistics_write_seconds +=
        std::chrono::duration<double>(Clock::now() - startWriteTime).count();
  }

  for (int indexTimeSample = 0; indexTimeSample < num_sample_;
       indexTimeSample++)
  {
    auto startComputeTime = Clock::now();
    // computeOneStep attend déjà la fin des kernels Kokkos avec FENCE.
    m_solver->computeOneStep(dt_, indexTimeSample, solverData);
    metrics_.compute_seconds +=
        std::chrono::duration<double>(Clock::now() - startComputeTime).count();
    
    if (is_in_situ_ && process_snapshots_ && indexTimeSample % snap_time_interval_ == 0)
    {
      const auto startAnalysisTime = Clock::now();
      // Read the solver's current buffer directly, including in Vector builds.
#ifdef USE_KOKKOS
      PressureExtrema extrema;
      if (extrema_workspace)
      {
        const auto measured = extrema_workspace->measure(solverData.m_pnGlobal, i1);
        extrema = measured.values;
        metrics_.extrema_prepare_seconds += measured.prepare_seconds;
        metrics_.extrema_compute_seconds += measured.compute_seconds;
        metrics_.extrema_result_seconds += measured.result_seconds;
        metrics_.extrema_total_seconds += measured.total_seconds;
        metrics_.extrema_input_bytes += measured.input_bytes;
        metrics_.extrema_result_bytes += measured.result_bytes;
        ++metrics_.extrema_calls;
      }
      else
        extrema = computePressureExtremaKokkos(solverData.m_pnGlobal, i1);
      const auto moments = computePressureMeanVarianceKokkos(solverData.m_pnGlobal, i1);
#else
      const auto extrema = computePressureExtremaCPU(solverData.m_pnGlobal, i1);
      const auto moments = computePressureMeanVarianceCPU(solverData.m_pnGlobal, i1);
#endif
      if (first_snapshot)
      {
        global_extrema = extrema;
      }
      else
      {
        global_extrema.minimum_pressure =
            std::min(global_extrema.minimum_pressure, extrema.minimum_pressure);
        global_extrema.maximum_pressure =
            std::max(global_extrema.maximum_pressure, extrema.maximum_pressure);
        global_extrema.maximum_absolute_pressure =
            std::max(global_extrema.maximum_absolute_pressure, extrema.maximum_absolute_pressure);
      }
      metrics_.snapshot_analysis_seconds +=
          std::chrono::duration<double>(Clock::now() - startAnalysisTime).count();

      // Stream reduced results instead of retaining a history of pressure fields.
      const auto startWriteTime = Clock::now();
      statistics << (first_snapshot ? "\n" : ",\n")
                 << "      {\n        \"timestep_index\": " << indexTimeSample << ",\n"
                 << "        \"time_seconds\": "
                 << (indexTimeSample + 1) * static_cast<double>(dt_) << ",\n"
                 << "        \"node_count\": " << m_mesh->getNumberOfNodes() << ",\n";
      writePressureStatistics(statistics, extrema, moments);
      statistics << "      }";
      if (!statistics) throw std::runtime_error("Cannot write in-situ statistics file");
      metrics_.statistics_write_seconds +=
          std::chrono::duration<double>(Clock::now() - startWriteTime).count();
      first_snapshot = false;
    }
    else if (is_snapshots_ && indexTimeSample % snap_time_interval_ == 0)
    {
      const auto startSnapshotTime = Clock::now();
      // Vector : le solveur possède une copie distincte du champ de pression.
      // La recopier ici pour que saveSnapshot lise les résultats calculés.
      // Kokkos : les vues partagent déjà les données, aucune copie n'est requise.
#ifdef USE_VECTOR
      pnGlobal = solverData.m_pnGlobal;
#endif
      saveSnapshot(indexTimeSample);
      // La préparation de la sortie (copie Vector incluse) est comptabilisée.
      metrics_.snapshot_write_seconds +=
          std::chrono::duration<double>(Clock::now() - startSnapshotTime)
              .count();
    }

    if (rle_ && process_snapshots_ && indexTimeSample % snap_time_interval_ == 0)
      compressSnapshotRLE(indexTimeSample, solverData.m_pnGlobal);

    if (indexTimeSample % 50 == 0)
    {
      m_solver->outputSolutionValues(indexTimeSample, i1, rhsElement[0],
                                     solverData.m_pnGlobal, "pnGlobal");
    }

    // Interpolation is common to both workflows; analysis is timed separately.
    startComputeTime = Clock::now();
    const int order = m_mesh->getOrder();

    for (int r = 0; r < static_cast<int>(receiver_coords_.size()); ++r)
    {
      float varnp1 = 0.0;
      for (int i = 0; i < order + 1; i++)
        {
      for (int j = 0; j < order + 1; j++)
      {
        for (int k = 0; k < order + 1; k++)
        {
          int nodeIdx = m_mesh->globalNodeIndex(rhsElementRcv[r], i, j, k);
          int globalNodeOnElement =
              i + j * (order + 1) + k * (order + 1) * (order + 1);
          varnp1 +=
              solverData.m_pnGlobal(nodeIdx, i1) *
              rhsWeightsRcv(r, globalNodeOnElement);
        }
      }
        }
      if (is_in_situ_)
        receiver_pressures[r] = varnp1;
      else
        pnAtReceiver(r, indexTimeSample) = varnp1;
    }

    if (is_in_situ_ && process_sismograms_)
    {
      metrics_.compute_seconds +=
          std::chrono::duration<double>(Clock::now() - startComputeTime).count();
      const auto startAnalysisTime = Clock::now();
      for (std::size_t r = 0; r < receiver_coords_.size(); ++r)
      {
        const double value = receiver_pressures[r];
        auto& extrema = receiver_extrema[r];
        if (receiver_moments[r].count == 0)
        {
          extrema = {value, value, std::abs(value)};
        }
        else
        {
          extrema.minimum_pressure = std::min(extrema.minimum_pressure, value);
          extrema.maximum_pressure = std::max(extrema.maximum_pressure, value);
          extrema.maximum_absolute_pressure =
              std::max(extrema.maximum_absolute_pressure, std::abs(value));
        }
        receiver_moments[r].update(value);
      }
      metrics_.sismogram_analysis_seconds +=
          std::chrono::duration<double>(Clock::now() - startAnalysisTime).count();
      startComputeTime = Clock::now();
    }

    swap(i1, i2);

    auto tmp = solverData.m_i1;
    solverData.m_i1 = solverData.m_i2;
    solverData.m_i2 = tmp;

    metrics_.compute_seconds +=
        std::chrono::duration<double>(Clock::now() - startComputeTime).count();
  }


  if (!is_in_situ_ && process_sismograms_)
  {
    const auto startSismogramTime = Clock::now();
    std::ofstream sismos("sismos.bin", std::ios::binary);
    if (!sismos) throw std::runtime_error("Cannot open sismos.bin");
    // Format natif : deux int32 (récepteurs, échantillons), puis pour chaque
    // échantillon des float32 (temps, pression de chaque récepteur).
    const std::int32_t dimensions[2] = {
        static_cast<std::int32_t>(receiver_coords_.size()),
        static_cast<std::int32_t>(num_sample_)};
    sismos.write(reinterpret_cast<const char*>(dimensions), sizeof(dimensions));
    for (int sample = 0; sample < num_sample_; ++sample)
    {
      const float time = (sample + 1) * dt_;
      sismos.write(reinterpret_cast<const char*>(&time), sizeof(time));
      for (int r = 0; r < static_cast<int>(receiver_coords_.size()); ++r)
      {
        const float pressure = pnAtReceiver(r, sample);
        sismos.write(reinterpret_cast<const char*>(&pressure), sizeof(pressure));
      }
    }
    sismos.close();
    if (!sismos) throw std::runtime_error("Cannot write sismos.bin");

    metrics_.sismogram_write_seconds =
        std::chrono::duration<double>(Clock::now() - startSismogramTime).count();
    metrics_.sismogram_mb =
        static_cast<double>(std::filesystem::file_size("sismos.bin")) / 1e6;

    if (rle_) compressSismogramsRLE();
  }
  else if (is_in_situ_)
  {
    std::vector<PressureMeanVariance> receiver_results;
    if (process_sismograms_)
    {
      const auto startAnalysisTime = Clock::now();
      receiver_results.reserve(receiver_coords_.size());
      for (const auto& state : receiver_moments)
        receiver_results.push_back(finalizePressureMeanVariance(state));
      metrics_.sismogram_analysis_seconds +=
          std::chrono::duration<double>(Clock::now() - startAnalysisTime).count();
    }

    const auto startWriteTime = Clock::now();
    if (process_snapshots_)
      statistics << "\n    ],\n    \"global_extrema\": {\n"
               << "      \"minimum_pressure\": " << global_extrema.minimum_pressure << ",\n"
               << "      \"maximum_pressure\": " << global_extrema.maximum_pressure << ",\n"
               << "      \"maximum_absolute_pressure\": "
               << global_extrema.maximum_absolute_pressure << "\n"
               << "    }\n  }";
    statistics << ",\n  \"sismograms\": ";
    if (process_sismograms_)
      statistics << "{\n    \"per_receiver\": [\n";
    else
      statistics << "null";
    for (std::size_t r = 0; r < receiver_coords_.size(); ++r)
    {
      if (r != 0) statistics << ",\n";
      statistics << "      {\n        \"receiver_index\": " << r << ",\n"
                 << "        \"sample_count\": " << receiver_moments[r].count << ",\n";
      writePressureStatistics(statistics, receiver_extrema[r], receiver_results[r]);
      statistics << "      }";
    }
    if (process_sismograms_)
      statistics << "\n    ]\n  }";
    statistics << "\n}\n";
    statistics.close();
    if (!statistics) throw std::runtime_error("Cannot write in-situ statistics file");
    metrics_.statistics_mb =
        static_cast<double>(std::filesystem::file_size(statistics_path)) / 1e6;
    metrics_.statistics_write_seconds +=
        std::chrono::duration<double>(Clock::now() - startWriteTime).count();
  }

  cout << "------------------------------------------------ " << endl;
  cout << "\n---- Elapsed Compute Time : " << metrics_.compute_seconds
       << " seconds." << endl;
  cout << "---- Elapsed Snapshot Write Time : "
       << metrics_.snapshot_write_seconds << " seconds." << endl;
  cout << "---- Elapsed Sismogram Write Time : "
       << metrics_.sismogram_write_seconds << " seconds."
       << endl;
  cout << "------------------------------------------------ " << endl;
}

// Initialize arrays
void SEMproxy::init_arrays()
{
  cout << "Allocate host memory for source and pressure values ..." << endl;

  rhsElement = allocateVector<vectorInt>(myNumberOfRHS, "rhsElement");
  rhsWeights = allocateArray2D<arrayReal>(
      myNumberOfRHS, m_mesh->getNumberOfPointsPerElement(), "RHSWeight");
  myRHSTerm = allocateArray2D<arrayReal>(myNumberOfRHS, num_sample_, "RHSTerm");
  pnGlobal =
      allocateArray2D<arrayReal>(m_mesh->getNumberOfNodes(), 2, "pnGlobal");
  if (!process_sismograms_) return;
  const int numberOfReceivers = static_cast<int>(receiver_coords_.size());
  if (!is_in_situ_)
    pnAtReceiver = allocateArray2D<arrayReal>(numberOfReceivers, num_sample_, "pnAtReceiver");
  // Receiver
  rhsElementRcv = allocateVector<vectorInt>(numberOfReceivers, "rhsElementRcv");
  rhsWeightsRcv = allocateArray2D<arrayReal>(
       numberOfReceivers, m_mesh->getNumberOfPointsPerElement(), "RHSWeightRcv");
}

// Initialize sources
void SEMproxy::init_source()
{
  arrayReal myRHSLocation = allocateArray2D<arrayReal>(1, 3, "RHSLocation");
  // std::cout << "All source are currently are coded on element 50." <<
  // std::endl;
  std::cout << "All source are currently are coded on middle element."
            << std::endl;
  int ex = nb_elements_[0];
  int ey = nb_elements_[1];
  int ez = nb_elements_[2];

  int lx = domain_size_[0];
  int ly = domain_size_[1];
  int lz = domain_size_[2];

  // Get source element index

  int source_index = floor((src_coord_[0] * ex) / lx) +
                     floor((src_coord_[1] * ey) / ly) * ex +
                     floor((src_coord_[2] * ez) / lz) * ey * ex;

  for (int i = 0; i < 1; i++)
  {
    rhsElement[i] = source_index;
  }

  // Get coordinates of the corners of the sourc element
  float cornerCoords[8][3];
  int I = 0;
  int nodes_corner[2] = {0, m_mesh->getOrder()};
  for (int k : nodes_corner)
  {
    for (int j : nodes_corner)
    {
      for (int i : nodes_corner)
      {
        int nodeIdx = m_mesh->globalNodeIndex(rhsElement[0], i, j, k);
        cornerCoords[I][0] = m_mesh->nodeCoord(nodeIdx, 0);
        cornerCoords[I][2] = m_mesh->nodeCoord(nodeIdx, 2);
        cornerCoords[I][1] = m_mesh->nodeCoord(nodeIdx, 1);
        I++;
      }
    }
  }

  // initialize source term
  vector<float> sourceTerm =
      myUtils.computeSourceTerm(num_sample_, dt_, f0, sourceOrder);
  for (int j = 0; j < num_sample_; j++)
  {
    myRHSTerm(0, j) = sourceTerm[j];
    if (j % 100 == 0)
      cout << "Sample " << j << "\t: sourceTerm = " << sourceTerm[j] << endl;
  }

  // get element number of source term
  myElementSource = rhsElement[0];
  cout << "Element number for the source location: " << myElementSource << endl
       << endl;

  int order = m_mesh->getOrder();

  switch (order)
  {
    case 1:
      SourceAndReceiverUtils::ComputeRHSWeights<1>(cornerCoords, src_coord_,
                                                   rhsWeights);
      break;
    case 2:
      SourceAndReceiverUtils::ComputeRHSWeights<2>(cornerCoords, src_coord_,
                                                   rhsWeights);
      break;
    case 3:
      SourceAndReceiverUtils::ComputeRHSWeights<3>(cornerCoords, src_coord_,
                                                   rhsWeights);
      break;
    default:
      throw std::runtime_error("Unsupported order: " + std::to_string(order));
  }

  if (!process_sismograms_) return;

  // Receiver computation
  // int receiver_index = floor((rcv_coord_[0] * ex) / lx) +
  //                      floor((rcv_coord_[1] * ey) / ly) * ex +
  //                      floor((rcv_coord_[2] * ez) / lz) * ey * ex;

  // for (int i = 0; i < 1; i++)
  // {
  //   rhsElementRcv[i] = receiver_index;
  // }

  // Receiver computation
  arrayReal receiverWeights = allocateArray2D<arrayReal>(
                                                         1, m_mesh->getNumberOfPointsPerElement(), "receiverWeights");

  for (int r = 0; r < static_cast<int>(receiver_coords_.size()); ++r)
    {
      const auto& coords = receiver_coords_[r];
      rhsElementRcv[r] = floor((coords[0] * ex) / lx) +
        floor((coords[1] * ey) / ly) * ex +
        floor((coords[2] * ez) / lz) * ey * ex;

      float cornerCoordsRcv[8][3];
      I = 0;
      for (int k : nodes_corner)
        {
          for (int j : nodes_corner)
            {
              for (int i : nodes_corner)
                {
                  int nodeIdx = m_mesh->globalNodeIndex(rhsElementRcv[r], i, j, k);
                  cornerCoordsRcv[I][0] = m_mesh->nodeCoord(nodeIdx, 0);
                  cornerCoordsRcv[I][1] = m_mesh->nodeCoord(nodeIdx, 1);
                  cornerCoordsRcv[I][2] = m_mesh->nodeCoord(nodeIdx, 2);
                  I++;
                }
            }
        }

      switch (order)
        {
        case 1:
          SourceAndReceiverUtils::ComputeRHSWeights<1>(
                                                       cornerCoordsRcv, coords, receiverWeights);
          break;
        case 2:
          SourceAndReceiverUtils::ComputeRHSWeights<2>(
                                                       cornerCoordsRcv, coords, receiverWeights);
          break;
        case 3:
          SourceAndReceiverUtils::ComputeRHSWeights<3>(
                                                       cornerCoordsRcv, coords, receiverWeights);
          break;
        default:
          throw std::runtime_error("Unsupported order: " + std::to_string(order));
        }

      for (int n = 0; n < m_mesh->getNumberOfPointsPerElement(); ++n)
        rhsWeightsRcv(r, n) = receiverWeights(0, n);
    }
}

SolverFactory::implemType SEMproxy::getImplem(string implemArg)
{
  if (implemArg == "makutu") return SolverFactory::MAKUTU;
  if (implemArg == "shiva") return SolverFactory::SHIVA;

  throw std::invalid_argument(
      "Implentation type does not follow any valid type.");
}

SolverFactory::meshType SEMproxy::getMesh(string meshArg)
{
  if (meshArg == "cartesian") return SolverFactory::Struct;
  if (meshArg == "ucartesian") return SolverFactory::Unstruct;

  std::cout << "Mesh type found is " << meshArg << std::endl;
  throw std::invalid_argument("Mesh type does not follow any valid type.");
}

SolverFactory::methodType SEMproxy::getMethod(string methodArg)
{
  if (methodArg == "sem") return SolverFactory::SEM;
  if (methodArg == "dg") return SolverFactory::DG;

  throw std::invalid_argument("Method type does not follow any valid type.");
}

float SEMproxy::find_cfl_dt(float cfl_factor)
{
  float sqrtDim3 = 1.73;  // to change for 2d
  float min_spacing = m_mesh->getMinSpacing();
  float v_max = m_mesh->getMaxSpeed();

  float dt = cfl_factor * min_spacing / (sqrtDim3 * v_max);

  return dt;
}

void SEMproxy::saveMesh()
{
  const auto path = std::filesystem::path(snap_folder_) / "mesh.bin";
  std::ofstream output(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("Cannot open mesh file: " + path.string());

  // Node order is shared by mesh.bin and every pressure snapshot.
  for (int node = 0; node < m_mesh->getNumberOfNodes(); ++node)
  {
    const float coordinates[3] = {m_mesh->nodeCoord(node, 0),
                                  m_mesh->nodeCoord(node, 1),
                                  m_mesh->nodeCoord(node, 2)};
    output.write(reinterpret_cast<const char*>(coordinates), sizeof(coordinates));
  }

  output.close();
  if (!output)
    throw std::runtime_error("Cannot write mesh file: " + path.string());
  // The first snapshot's timer includes this one-time geometry write.
  metrics_.snapshot_mb +=
      static_cast<double>(std::filesystem::file_size(path)) / 1e6;
}

void SEMproxy::saveSnapshot(int timestep)
{
  if (!mesh_saved_)
  {
    saveMesh();
    mesh_saved_ = true;
  }

  const auto path = std::filesystem::path(snap_folder_) /
    ("snapshot_" + std::to_string(timestep) + ".bin");
  std::ofstream output(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("Cannot open snapshot file: " + path.string());

  for (int node = 0; node < m_mesh->getNumberOfNodes(); ++node)
    {
      const float pressure = pnGlobal(node, i1);
      output.write(reinterpret_cast<const char*>(&pressure), sizeof(pressure));
    }
  
  output.close();
  if (!output)
    throw std::runtime_error("Cannot write snapshot file: " + path.string());
  // Compter uniquement les fichiers sauvegardés pendant cette exécution.
  metrics_.snapshot_mb +=
      static_cast<double>(std::filesystem::file_size(path)) / 1e6;
}


void SEMproxy::compressSnapshotRLE(int timestep, const arrayReal& pressure)
{
  using Clock = std::chrono::steady_clock;
  const int nbNodes = m_mesh->getNumberOfNodes();

  // 1. Copy the newest field (column i1) into a contiguous host buffer.
  //    computeOneStep() ends with a fence, so the GPU work is finished.
  auto start = Clock::now();
  rle_buffer_.resize(nbNodes);
  for (int node = 0; node < nbNodes; ++node) rle_buffer_[node] = pressure(node, i1);
  const double copySeconds =
      std::chrono::duration<double>(Clock::now() - start).count();

  // 2. Encode.
  start = Clock::now();
  const auto runs = rleEncode(rle_buffer_.data(), rle_buffer_.size());
  const double encodeSeconds =
      std::chrono::duration<double>(Clock::now() - start).count();

  // 3. Write the compressed field.
  start = Clock::now();
  const auto path = std::filesystem::path(rle_folder_) /
                    ("snapshot_" + std::to_string(timestep) + ".rle");
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot open " + path.string());
  rleWrite(out, runs);
  out.close();
  if (!out) throw std::runtime_error("Cannot write " + path.string());
  const double writeSeconds =
      std::chrono::duration<double>(Clock::now() - start).count();

  // 4. Record one line per snapshot (ratio over time) and the totals.
  const std::size_t rawBytes = rle_buffer_.size() * sizeof(float);
  const std::size_t rleBytesWritten = std::filesystem::file_size(path);
  std::ofstream csv(std::filesystem::path(rle_folder_) / "rle_snapshots.csv",
                    std::ios::app);
  csv << std::setprecision(9) << timestep << ',' << (timestep + 1) * dt_ << ','
      << nbNodes << ',' << runs.size() << ',' << rawBytes << ','
      << rleBytesWritten << ',' << double(rawBytes) / rleBytesWritten << ','
      << copySeconds << ',' << encodeSeconds << ',' << writeSeconds << '\n';

  ++metrics_.rle_snapshot_count;
  metrics_.rle_copy_seconds += copySeconds;
  metrics_.rle_encode_seconds += encodeSeconds;
  metrics_.rle_write_seconds += writeSeconds;
  metrics_.rle_snapshot_raw_mb += rawBytes / 1e6;
  metrics_.rle_snapshot_mb += rleBytesWritten / 1e6;
}

void SEMproxy::compressSismogramsRLE()
{
  using Clock = std::chrono::steady_clock;
  const int nbReceivers = static_cast<int>(receiver_coords_.size());

  std::ofstream csv(std::filesystem::path(rle_folder_) / "rle_sismograms.csv");
  csv << "receiver,values,runs,raw_bytes,rle_bytes,ratio,encode_s\n";
  const auto path = std::filesystem::path(rle_folder_) / "sismos.rle";
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot open " + path.string());

  std::vector<float> signal(num_sample_);
  std::size_t rawBytes = 0;
  for (int r = 0; r < nbReceivers; ++r)
  {
    for (int s = 0; s < num_sample_; ++s) signal[s] = pnAtReceiver(r, s);

    auto start = Clock::now();
    const auto runs = rleEncode(signal.data(), signal.size());
    const double encodeSeconds =
        std::chrono::duration<double>(Clock::now() - start).count();

    start = Clock::now();
    rleWrite(out, runs);  // one block per receiver, in receiver order
    metrics_.rle_sismogram_write_seconds +=
        std::chrono::duration<double>(Clock::now() - start).count();

    const std::size_t raw = signal.size() * sizeof(float);
    const std::size_t encoded = sizeof(std::uint64_t) + rleBytes(runs);
    rawBytes += raw;
    metrics_.rle_sismogram_encode_seconds += encodeSeconds;
    csv << std::setprecision(9) << r << ',' << signal.size() << ','
        << runs.size() << ',' << raw << ',' << encoded << ','
        << double(raw) / encoded << ',' << encodeSeconds << '\n';
  }
  out.close();
  if (!out) throw std::runtime_error("Cannot write " + path.string());
  metrics_.rle_sismogram_raw_mb = rawBytes / 1e6;
  metrics_.rle_sismogram_mb = std::filesystem::file_size(path) / 1e6;
}
