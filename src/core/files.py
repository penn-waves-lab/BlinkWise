from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectFiles:
    """
    This class contains the file names for the project.
    """
    # ==============================
    # raw data related
    # ==============================
    metadata = Path('metadata.json')
    # radar data related
    radar_raw_data_filename = Path('radar_data.npy')
    radar_timestamps_filename = Path('radar_timestamps.npy')
    # video data related
    video_label_dataframe_filename = Path('label.csv')
    video_timestamps_filename = Path('video_timestamps.npy')
    # visual stimuli related
    stimuli_log_filename = Path('stimuli.log')
    # denied regions
    deny_list_filename = Path('deny_list.json')
    deny_file_list = Path('deny_file_list.json')

    # ==============================
    # processed data related / dataset
    # ==============================
    dataset_processing_protocol_filename = Path('processing_protocol.txt')
    dataset_processing_config_filename = Path('processing_config.json')

    model_input_filename = Path('input.npy')
    model_label_filename = Path('label.npz')
    model_target_ratio_filename = Path('target_ratio.npz')

    rf_deny_list_filename = Path('rf_deny_list.json')


project_files = ProjectFiles()