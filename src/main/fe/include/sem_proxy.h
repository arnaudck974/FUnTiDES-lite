//************************************************************************
//   proxy application v.0.0.1
//
//  semproxy.hpp: the main interface of SEM proxy application
//
//************************************************************************

#ifndef SEMPROXY_HPP_
#define SEMPROXY_HPP_

#include <data_type.h>
#include <model_struct.h>
#include <model_unstruct.h>
#include <solver_factory.h>
#include <utils.h>
#include <vector>

#include <memory>
#include <string>
#include <variant>

#include "sem_proxy_options.h"

// Temps en secondes ; volumes en MB (1 MB = 1 000 000 octets).
struct SemProxyMetrics
{
  double total_seconds = 0;
  double compute_seconds = 0;
  double snapshot_write_seconds = 0;
  double sismogram_write_seconds = 0;
  double snapshot_mb = 0;
  double sismogram_mb = 0;
  bool in_situ = false;
  std::string outputs = "both";
  int sample_count = 0;
  std::string extrema_backend;
  std::string execution_space;
  double extrema_prepare_seconds = 0, extrema_compute_seconds = 0;
  double extrema_result_seconds = 0, extrema_total_seconds = 0;
  std::size_t extrema_input_bytes = 0, extrema_result_bytes = 0, extrema_calls = 0;
  double snapshot_analysis_seconds = 0;
  double sismogram_analysis_seconds = 0;
  double statistics_write_seconds = 0;
  double statistics_mb = 0;
  bool rle = false;
  int rle_snapshot_count = 0;
  double rle_copy_seconds = 0, rle_encode_seconds = 0, rle_write_seconds = 0;
  double rle_snapshot_raw_mb = 0, rle_snapshot_mb = 0;
  double rle_sismogram_encode_seconds = 0, rle_sismogram_write_seconds = 0;
  double rle_sismogram_raw_mb = 0, rle_sismogram_mb = 0;
};

/**
 * @class SEMproxy
 */

class SEMproxy
{
 public:
  /**
   * @brief Constructor of the SEMproxy class
   */
  SEMproxy(const SemProxyOptions& cfg);

  /**
   * @brief Destructor of the SEMproxy class
   */
  ~SEMproxy(){};

  /**
   * @brief Initialize the simulation.
   * @post run()
   */
  void initFiniteElem()
  {
    init_arrays();
    init_source();
  };

  /**
   * @brief Run the simulation.
   * @pre This must be called after init()
   * @post Optional printout performance resutls
   */
  void run();

  const SemProxyMetrics& getMetrics() const { return metrics_; }

  /**
   * @brief Computes optimal time step using CFL stability condition for seismic
   * wave propagation
   *
   * Calculates the maximum stable time step for explicit finite difference
   * schemes using the CFL condition: dt ≤ CFL_factor × min_spacing / (√D ×
   * v_max) where D is the number of dimension.
   *
   * @param cfl_factor Stability factor (0.5-0.7 for 2nd-order, 0.3-0.4 for
   * higher-order schemes)
   * @return dt the max timestep.
   *
   * @note Typical values: 0.5-0.7 for 2nd-order, 0.3-0.4 for higher-order
   * schemes
   * @warning Must be called before time-stepping loop to ensure numerical
   * stability
   *
   */
  float find_cfl_dt(float cfl_factor);

  void saveSnapshot(int timestep);

  /**
   * @brief Compresses the current pressure field with RLE (TP3).
   *
   * Copies the newest field into a contiguous host buffer, encodes it,
   * writes <rle_folder>/snapshot_<step>.rle and appends one line to
   * <rle_folder>/rle_snapshots.csv. Copy, encoding and writing are timed
   * separately in the metrics.
   */
  void compressSnapshotRLE(int timestep, const arrayReal& pressure);

  /// @brief Compresses each receiver signal with RLE (ad-hoc mode only).
  void compressSismogramsRLE();
 private:
  void saveMesh();

  SemProxyMetrics metrics_;
  bool is_in_situ_ = false;
  std::string extrema_backend_;
  bool process_snapshots_ = true;
  bool process_sismograms_ = true;
  int i1 = 0;
  int i2 = 1;

  // proper to cartesian mesh
  // or any structured mesh
  int nb_elements_[3] = {0};
  int nb_nodes_[3] = {0};
  std::array<float, 3> src_coord_ = {0};
  std::array<float, 3> rcv_coord_ = {0};

  std::vector<std::array<float, 3>> receiver_coords_;
  float domain_size_[3] = {0};
  
  // snapshots
  bool is_snapshots_;
  bool mesh_saved_ = false;
  int snap_time_interval_;
  std::string snap_folder_;

  bool rle_ = false;
  std::string rle_folder_;
  std::vector<float> rle_buffer_;

  // physics
  bool isElastic_;

  // time parameters
  float dt_;
  float timemax_;
  int num_sample_;
  // source parameters
  const int myNumberOfRHS = 1;
  const float f0 = 5.;
  const int sourceOrder = 2;
  int myElementSource = 0;

  std::shared_ptr<model::ModelApi<float, int>> m_mesh;
  std::unique_ptr<SEMSolverBase> m_solver;
  SolverUtils myUtils;

  // arrays
  arrayReal myRHSTerm;
  arrayReal pnGlobal;
  vectorInt rhsElement;
  vectorInt rhsElementRcv;
  arrayReal rhsWeights;
  arrayReal rhsWeightsRcv;
  arrayReal pnAtReceiver;

  // initialize source and RHS
  void init_source();

  // allocate arrays and vectors
  void init_arrays();

  // private methods to pars argv options
  int getPhysic(string physicArg);
  SolverFactory::implemType getImplem(string implemArg);
  SolverFactory::methodType getMethod(string methodArg);
  SolverFactory::meshType getMesh(string meshArg);
};

#endif /* SEMPROXY_HPP_ */
