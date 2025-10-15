import os
import polars as pl
import pyarrow.parquet as pq
import methylcheck
import yaml
import logging
import time
from pathlib import Path

# --- 1. LOGGING SYSTEM SETUP  ---
Path("/app/results").mkdir(exist_ok=True)
log_filename = f"/app/results/filtering_run_{time.strftime('%Y%m%d-%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# --- End of Logging Setup ---


# --- 2. HELPER FUNCTIONS  ---
def report_filtering_step(step_name, before_shape, after_shape, details=None):
    """Consistent reporting function using the logger."""
    logger.info(f"--- Filtering step: {step_name} ---")
    logger.info(f"Before: {before_shape[0]} samples × {before_shape[1]} probes")
    logger.info(f"After:  {after_shape[0]} samples × {after_shape[1]} probes")
    logger.info(f"Removed: {before_shape[0] - after_shape[0]} samples, {before_shape[1] - after_shape[1]} probes")
    
    if details:
        for key, value in details.items():
            logger.info(f"    {key}: {value}")
    logger.info("-" * (len(step_name) + 24))


def filter_snp_adjacent(df: pl.DataFrame) -> pl.DataFrame:
    """Remove probes adjacent to SNPs."""
    initial_shape = df.shape
    sketchy_probe_list = methylcheck.list_problem_probes('epic', ["Polymorphism"])
    good_probes = [col for col in df.columns if col not in sketchy_probe_list]
    filtered_df = df.select(good_probes)
    report_filtering_step("SNP Adjacent Probe Filtering", initial_shape, filtered_df.shape, {"SNP adjacent probes identified": len(sketchy_probe_list)})
    return filtered_df


def filter_problem_samples(df: pl.DataFrame) -> pl.DataFrame:
    """Remove samples with too many missing values."""
    initial_shape = df.shape
    na_counts = df.select([pl.sum_horizontal(pl.all().is_null()).alias("na_count")])
    samples_to_keep = df.with_row_index("idx").filter(na_counts.select("na_count").to_series() <= 50000).select("idx").to_series().to_list()
    filtered_df = df.select(pl.all()).filter(pl.arange(0, df.height).is_in(samples_to_keep))
    report_filtering_step("Problem Sample Filtering", initial_shape, filtered_df.shape, {"Samples with >50k missing values": initial_shape[0] - filtered_df.shape[0]})
    return filtered_df


def filter_cross_hybrid(df: pl.DataFrame) -> pl.DataFrame:
    """Remove cross-hybridizing probes."""
    initial_shape = df.shape
    sketchy_probe_list = methylcheck.list_problem_probes('epic', ["CrossHybridization"])
    good_probes = [col for col in df.columns if col not in sketchy_probe_list]
    filtered_df = df.select(good_probes)
    report_filtering_step("Cross-Hybridization Probe Filtering", initial_shape, filtered_df.shape, {"Cross-hybridizing probes": len(sketchy_probe_list)})
    return filtered_df


# --- 3. MODIFIED MAIN FUNCTION  ---
def main():
    """Main function to process the data."""
    logger.info("📊 Starting Methylation Data Filtering Process 📊")
    logger.warning("--- RUNNING IN TEMPORARY MODE: Steps requiring probe metadata are SKIPPED. ---")
    
    try:
        config_path = Path("/app/config/filtering_config.yaml")
        if not config_path.is_file():
            logger.error(f"Configuration file not found! Please ensure '{config_path}' exists.")
            return

        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        m_values_file_path = config["base_m_value_file"]
        m_values_filtered_data_file = config["filtered_m_value_file"]
        # probe_metadata_file = config["probe_metadata_file"] # Temporarily disabled
        
        logger.info(f"Input M-values file: {m_values_file_path}")
        # logger.info(f"Input probe metadata file: {probe_metadata_file}") # Temporarily disabled
        logger.info(f"Output filtered file: {m_values_filtered_data_file}")

        if not Path(m_values_file_path).is_file():
            logger.error(f"Error: Base m-values file not found at {m_values_file_path}")
            return
        
        # ==================== TEMPORARY MODIFICATION: Skip metadata-related steps ====================
        # if not Path(probe_metadata_file).is_file():
        #     logger.error(f"Error: Probe metadata file not found at {probe_metadata_file}")
        #     return
        # logger.info("Loading probe metadata...")
        # probe_metadata = pl.read_csv(probe_metadata_file)
        # logger.info("Identifying gene-associated probes...")
        # gene_associated_probes = probe_metadata.filter(pl.col("gencodebasic_name").is_not_null()).select("probe_id").to_series().to_list()
        # logger.info("Identifying XYM probes...")
        # xym_probes = probe_metadata.filter(pl.col("chr_hg38").is_in(["chrX", "chrY", "chrM"])).select("probe_id").to_series().to_list()
        # =============================================================================================

        logger.info("Reading base data table...")
        df = pl.read_csv(m_values_file_path)
        initial_shape = df.shape
        logger.info(f"Initial data shape: {initial_shape[0]} samples × {initial_shape[1]} probes")
        
        # ==================== TEMPORARY MODIFICATION: Skip metadata-related filtering ====================
        # logger.info("Subsetting to gene-associated probes...")
        # available_gene_probes = [probe for probe in gene_associated_probes if probe in df.columns]
        # df = df.select(["sampleId"] + available_gene_probes) if "sampleId" in df.columns else df.select(available_gene_probes)
        # report_filtering_step("Gene-Associated Probe Filtering", initial_shape, df.shape, {"Gene-associated probes kept": len(available_gene_probes)})
        
        # logger.info("Removing XYM probes...")
        # pre_xym_shape = df.shape
        # available_xym_probes = [probe for probe in xym_probes if probe in df.columns]
        # columns_to_keep = [col for col in df.columns if col not in available_xym_probes]
        # df = df.select(columns_to_keep)
        # report_filtering_step("XYM Probe Filtering", pre_xym_shape, df.shape, {"XYM probes removed": len(available_xym_probes)})
        # ===============================================================================================

        # === These core filtering steps, which do not depend on metadata, will run normally! ===
        logger.info("Starting core filtering steps that do not require metadata...")
        df = filter_problem_samples(df)
        df = filter_snp_adjacent(df)
        df = filter_cross_hybrid(df)
        
        logger.info("Saving filtered m-values...")
        df.write_parquet(m_values_filtered_data_file)
        logger.info(f"✅ M-values saved to {m_values_filtered_data_file}")
        
        logger.info("🎉 Filtering process (temporary mode) completed successfully! 🎉")

    except Exception as e:
        logger.critical("A fatal error occurred, and the process was interrupted!")
        logger.exception(e) # This will log the full error traceback

if __name__ == "__main__":
    main()