//************************************************************************
//  SEM proxy application v.0.0.1
//
//  main.cpp: this main file is simply a driver
//************************************************************************

#include "sem_proxy.h"
#include "sem_proxy_options.h"

#include <fstream>
#include <iomanip>

time_point<steady_clock> startInitTime;

static void saveMetrics(const SemProxyMetrics& metrics)
{
  std::ofstream output("metrics.json");
  if (!output) throw std::runtime_error("Cannot open metrics.json");
  // Un rapport par essai ; une nouvelle simulation réinitialise Python à null.
  output << std::setprecision(17)
         << "{\n  \"simulation\": {\n"
         << "    \"outputs\": \"" << metrics.outputs << "\",\n"
         << "    \"sample_count\": " << metrics.sample_count << ",\n"
         << "    \"total_seconds\": " << metrics.total_seconds << ",\n"
         << "    \"compute_seconds\": " << metrics.compute_seconds << ",\n"
         << "    \"snapshot_write_seconds\": "
         << metrics.snapshot_write_seconds << ",\n"
         << "    \"sismogram_write_seconds\": "
         << metrics.sismogram_write_seconds << ",\n"
         << "    \"snapshot_mb\": " << metrics.snapshot_mb << ",\n"
         << "    \"sismogram_mb\": " << metrics.sismogram_mb << "\n"
         << "  },\n"
         << R"json(  "ad_hoc": {
    "statistics": {
      "snapshots": {"read_seconds": null, "analysis_seconds": null},
      "sismograms": {"read_seconds": null, "analysis_seconds": null},
      "output_file": {"write_seconds": null, "size_mb": null}
    },
    "visualization": {
      "snapshots": {"read_seconds": null, "preparation_seconds": null, "render_seconds": null},
      "sismograms": {"read_seconds": null, "preparation_seconds": null, "render_seconds": null}
    }
  },
  "in_situ": )json";
  if (metrics.in_situ)
  {
    output << "{\n"
           << "    \"snapshots\": {\"analysis_seconds\": "
           << metrics.snapshot_analysis_seconds << "},\n"
           << "    \"sismograms\": {\"analysis_seconds\": "
           << metrics.sismogram_analysis_seconds << "},\n"
           << "    \"output_file\": {\"write_seconds\": "
           << metrics.statistics_write_seconds << ", \"size_mb\": "
           << metrics.statistics_mb << "}\n"
           << "  }";
  }
  else
  {
    output << "null";
  }
  output << ",\n  \"extrema_comparison\": ";
  if (metrics.extrema_backend.empty()) output << "null";
  else
    output << "{\"backend\": \"" << metrics.extrema_backend
           << "\", \"execution_space\": \"" << metrics.execution_space
           << "\", \"calls\": " << metrics.extrema_calls
           << ", \"prepare_seconds\": " << metrics.extrema_prepare_seconds
           << ", \"compute_seconds\": " << metrics.extrema_compute_seconds
           << ", \"result_seconds\": " << metrics.extrema_result_seconds
           << ", \"total_seconds\": " << metrics.extrema_total_seconds
           << ", \"input_bytes\": " << metrics.extrema_input_bytes
           << ", \"result_bytes\": " << metrics.extrema_result_bytes << "}";
  output << ",\n  \"rle\": ";
  if (!metrics.rle) output << "null";
  else
    output << "{\"snapshot_count\": " << metrics.rle_snapshot_count
           << ", \"copy_seconds\": " << metrics.rle_copy_seconds
           << ", \"encode_seconds\": " << metrics.rle_encode_seconds
           << ", \"write_seconds\": " << metrics.rle_write_seconds
           << ", \"snapshot_raw_mb\": " << metrics.rle_snapshot_raw_mb
           << ", \"snapshot_mb\": " << metrics.rle_snapshot_mb
           << ", \"sismogram_encode_seconds\": "
           << metrics.rle_sismogram_encode_seconds
           << ", \"sismogram_write_seconds\": "
           << metrics.rle_sismogram_write_seconds
           << ", \"sismogram_raw_mb\": " << metrics.rle_sismogram_raw_mb
           << ", \"sismogram_mb\": " << metrics.rle_sismogram_mb << "}";
  output << "\n}\n";
  output.close();
  if (!output) throw std::runtime_error("Cannot write metrics.json");
}

void compute(SEMproxy &semsim)
{
  cout << "\n+================================= " << endl;
  cout << "| Running SEM Application ...      " << endl;
  cout << "+================================= \n" << endl;

  // start timer
  time_point<steady_clock> startRunTime = steady_clock::now();
  semsim.run();

  cout << "\n+================================= " << endl;
  cout << "| SEM Application Finished.       " << endl;
  cout << "+================================= \n" << endl;

  // print timing information
  cout << "Elapsed Initial Time : "
       << duration<double>(startRunTime - startInitTime).count()
       << " seconds." << endl;
  cout << "Elapsed Run Time : "
       << duration<double>(steady_clock::now() - startRunTime).count()
       << " seconds."
       << endl;
};

void compute_loop(SEMproxy &semsim) { compute(semsim); }

int main(int argc, char *argv[])
{
  startInitTime = steady_clock::now();
  SemProxyMetrics metrics;

#ifdef USE_KOKKOS
  // Keep caller-defined affinity settings, including those used for scaling runs.
  setenv("OMP_PROC_BIND", "spread", 0);
  setenv("OMP_PLACES", "threads", 0);
  Kokkos::initialize(argc, argv);
  {
#endif

    cxxopts::Options options("SEM Proxy", "Runs the SEM simulation.");
    options.allow_unrecognised_options();  // lets Kokkos flags pass

    options.add_options()("h,help", "Print help message");

    SemProxyOptions opt;
    SemProxyOptions::bind_cli(options, opt);

    auto result = options.parse(argc, argv);

#ifdef USE_KOKKOS
    // Report the initialized backend and host pool, rather than requested values.
    std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
    std::cout << "Kokkos host threads: " << Kokkos::DefaultHostExecutionSpace().concurrency() << std::endl;
#endif

    if (result.count("help"))
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    try
    {
      opt.validate();
    }
    catch (const std::exception &e)
    {
      // your error path (no help printing here)
      std::cerr << "Invalid options: " << e.what() << "\n";
      return 1;
    }

    cout << "+==================================+" << endl;
    cout << "| Initializing SEM Application ... |" << endl;
    cout << "+==================================+\n" << endl;

    SEMproxy semsim(opt);

    compute_loop(semsim);
    metrics = semsim.getMetrics();

#ifdef USE_KOKKOS
  }
  Kokkos::finalize();
#endif

  // Arrêter le temps total avant l'écriture du rapport de mesure lui-même.
  metrics.total_seconds =
      duration<double>(steady_clock::now() - startInitTime).count();
  saveMetrics(metrics);
  cout << "Elapsed TotalExe Time : "
       << metrics.total_seconds << " seconds.\n"
       << endl;
  return (0);
}
