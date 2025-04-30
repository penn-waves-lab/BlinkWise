import json
import os
import warnings
from typing import Sequence

from ..config.experiment_config import ExperimentConfig


def get_first_experiment_per_subject(base_folder, experiment_list):
    """
    Retrieves the first experiment for each unique subject from a list of experiments.

    Args:
        base_folder (str): The base directory path where all experiment folders are located.
        experiment_list (List[str]): A list of experiment names (folder names) to process.

    Returns:
        dict[str, str]: A dictionary where keys are subject names and values
        are the names of the first experiment for that subject.
    """
    subject_first_experiment = {}

    for experiment in experiment_list:
        exp_folder = os.path.join(base_folder, experiment)
        metadata_file = os.path.join(exp_folder, 'metadata.json')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            subject = metadata.get('subject', 'unknown').lower()

            if subject not in subject_first_experiment:
                subject_first_experiment[subject] = experiment

    return subject_first_experiment


def summarize_subjects_and_experiments(base_folder, exp_folders) -> dict[str, list[str]]:
    """
    organize the experiment folders by subject names.

    Args:
        base_folder (str): The base directory path where all experiment folders are located.
        exp_folders (List[str]): A list of experiment names (folder names) to process.

    Returns:
        dict[str, list[str]]: A dictionary where keys are subject names and values are the list of experiment names.
    """
    # construct a dict with subject names as keys and the list of experiment names as values
    subject_exp_dict = {}
    for exp_folder in exp_folders:
        metadata_file = os.path.join(base_folder, exp_folder, 'metadata.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        subject = metadata.get('subject', 'unknown').strip().lower().replace(" ", "")
        if subject not in subject_exp_dict:
            subject_exp_dict[subject] = []
        subject_exp_dict[subject].append(exp_folder)

    # sort the list of experiment folders
    for subject in subject_exp_dict:
        subject_exp_dict[subject] = list(sorted(subject_exp_dict[subject]))
    # sort the dictionary by subject name
    subject_exp_dict = dict(sorted(subject_exp_dict.items()))

    return subject_exp_dict


def load_exp_subset(base_folder, subjects_to_exclude: Sequence[str] = (), verbose: bool = True) -> list[str]:
    """
    Load the list of experiment folders to include in the analysis.

    Args:
        base_folder (str): The base directory path where all experiment folders are located.
        subjects_to_exclude (Sequence[str]): A list of subject names to exclude from the analysis. Empty includes all subjects.
        verbose (bool): Whether to print the included subjects and their experiments.

    Returns:
        list[str]: A list of experiment names (folder names) to include in the analysis.
    """

    all_exp_folders = list(sorted(list(filter(lambda x: "exp" in x, os.listdir(base_folder)))))

    exp_subset_file_path = os.path.join(base_folder, "deny_file_list.json")
    if not os.path.exists(exp_subset_file_path):
        exp_folders_after_denied = all_exp_folders
    else:
        with open(exp_subset_file_path, 'r') as f:
            deny_file_list = json.load(f)
        exp_folders_after_denied = list(filter(lambda x: x not in deny_file_list, all_exp_folders))

        if verbose:
            print(f"Excluded experiments from denied file list:")
            for exp in deny_file_list:
                print(f"  - {exp}")

    # construct a dict with subject names as keys and the list of experiment names as values
    subject_exp_dict = summarize_subjects_and_experiments(base_folder, exp_folders_after_denied)

    # exclude the subject if provided
    if len(subjects_to_exclude) > 0:
        if verbose:
            print(f"Excluding subject: {'; '.join(subjects_to_exclude)}")

        # subjects to exclude is provided
        subjects_to_exclude = list(map(lambda s: s.strip().lower().replace(" ", ""), subjects_to_exclude))

        # remove the subject to exclude from the dictionary
        for subject_name in subjects_to_exclude:
            if subject_name in subject_exp_dict:
                subject_exp_dict.pop(subject_name)

        # construct the list of experiment folders
        filtered_exp_folders = []
        for exp_list in subject_exp_dict.values():
            filtered_exp_folders += exp_list
    else:
        filtered_exp_folders = exp_folders_after_denied

    # print the included subjects and their experiments
    if verbose:
        print("Included subjects and their experiments:")
        for subject, exp_list in sorted(subject_exp_dict.items()):
            print(f"  - {subject}: {list(sorted(exp_list))}")

    # sort the list of experiment folders
    filtered_exp_folders = list(sorted(filtered_exp_folders))

    return filtered_exp_folders


def prepare_exp_subsets(exp_config: ExperimentConfig, verbose=True) -> tuple[Sequence[str], Sequence[str]]:
    """
    Prepare the training and validation sets for the experiment based on the provided configuration.
    """
    if exp_config.leave_one_out_subject is not None:
        # if leave_one_out_subject is set,:
        # (1) we remove the leave_one_out_subject from the training set;
        # (2) the validation set will be the data from the leave_one_out_subject.
        print(f"Leave one out subject: {exp_config.leave_one_out_subject}")
        subjects_to_exclude = exp_config.subjects_to_exclude + [exp_config.leave_one_out_subject]
        exp_subset = load_exp_subset(exp_config.data_folder, subjects_to_exclude=subjects_to_exclude, verbose=verbose)

        # Load the validation set
        all_included_subjects = summarize_subjects_and_experiments(exp_config.data_folder, exp_subset).keys()
        val_subjects_to_exclude = list(exp_config.subjects_to_exclude) + list(all_included_subjects)
        val_exp_subset = load_exp_subset(exp_config.data_folder, subjects_to_exclude=val_subjects_to_exclude,
                                         verbose=verbose)
    else:
        subjects_to_exclude = exp_config.subjects_to_exclude
        val_subjects_to_exclude = exp_config.subjects_to_exclude

        exp_subset = load_exp_subset(exp_config.data_folder, subjects_to_exclude=subjects_to_exclude, verbose=verbose)
        val_exp_subset = load_exp_subset(exp_config.data_folder, subjects_to_exclude=val_subjects_to_exclude,
                                         verbose=False)

    return exp_subset, val_exp_subset
