#!/usr/bin/env python

import argparse
import os
import sys

import pandas as pd

from evaluate_model import evaluate_model
from helper_code import DEMOGRAPHICS_FILE, HEADERS, find_patients, update_demographics_table
from team_code import load_model as load_team_model
from team_code import run_model as run_team_model
from team_code import train_model as train_team_model


def get_parser() -> argparse.ArgumentParser:
    description = 'Train, run, and optionally evaluate the Cox pipeline in one command.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--train-data', type=str, help='Folder with labeled training data.')
    parser.add_argument('--test-data', type=str, help='Folder with data to score, such as supplementary_set.')
    parser.add_argument('--model-folder', type=str, required=True, help='Folder containing or receiving model.sav.')
    parser.add_argument('--output-folder', type=str, help='Folder for demographics predictions when scoring a dataset.')
    parser.add_argument('--score-file', type=str, help='Optional score file written when test labels are available.')
    parser.add_argument('--csv-path', type=str, default=None, help='Optional channel table override.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-f', '--allow-failures', action='store_true')
    return parser


def get_demographics_path(data_folder: str) -> str:
    return os.path.join(data_folder, DEMOGRAPHICS_FILE)


def has_column(data_folder: str, column_name: str) -> bool:
    demographics_file = get_demographics_path(data_folder)
    if not os.path.isfile(demographics_file):
        raise FileNotFoundError(f'Missing demographics file: {demographics_file}')

    columns = pd.read_csv(demographics_file, nrows=0).columns
    return column_name in columns


def train_if_requested(args: argparse.Namespace) -> None:
    if not args.train_data:
        return

    if args.verbose:
        print(f'Training Cox pipeline from: {args.train_data}')

    if args.csv_path is None:
        train_team_model(args.train_data, args.model_folder, args.verbose)
    else:
        train_team_model(args.train_data, args.model_folder, args.verbose, csv_path=args.csv_path)


def run_inference(data_folder: str, model_folder: str, output_folder: str, verbose: bool, allow_failures: bool) -> str:
    if verbose:
        print(f'Loading model from: {model_folder}')

    model = load_team_model(model_folder, verbose)
    patient_metadata_list = find_patients(get_demographics_path(data_folder))
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise ValueError(f'No records found in: {data_folder}')

    os.makedirs(output_folder, exist_ok=True)

    if verbose:
        print(f'Running inference on {num_records} records from: {data_folder}')

    all_results: dict[str, tuple[int | float, float]] = {}
    for index, record in enumerate(patient_metadata_list, start=1):
        patient_id = str(record[HEADERS['bids_folder']])
        session_id = str(record[HEADERS['session_id']])

        if verbose:
            width = len(str(num_records))
            print(f'- {index:>{width}}/{num_records}: {patient_id} (Session {session_id})')

        try:
            binary_output, probability_output = run_team_model(model, record, data_folder, verbose)
        except Exception:
            if not allow_failures:
                raise
            if verbose:
                print('  prediction failed; writing NaNs because --allow-failures was set')
            binary_output, probability_output = float('nan'), float('nan')

        all_results[patient_id] = (binary_output, probability_output)

    output_table_path = update_demographics_table(get_demographics_path(data_folder), output_folder, all_results)

    if verbose:
        print(f'Predictions written to: {output_table_path}')

    return output_table_path


def evaluate_if_possible(test_data: str, predictions_file: str, score_file: str | None, verbose: bool) -> None:
    if not has_column(test_data, HEADERS['label']):
        if verbose:
            print('Skipping evaluation because the test demographics file has no label column.')
        return

    labels_file = get_demographics_path(test_data)
    auroc, auprc, accuracy, f_measure = evaluate_model(labels_file, predictions_file)
    output_string = (
        f'AUROC: {auroc:.3f}\n'
        f'AUPRC: {auprc:.3f}\n'
        f'Accuracy: {accuracy:.3f}\n'
        f'F-measure: {f_measure:.3f}\n'
    )

    if score_file:
        with open(score_file, 'w') as file_handle:
            file_handle.write(output_string)

    if verbose or not score_file:
        print(output_string, end='')


def run(args: argparse.Namespace) -> None:
    if not args.train_data and not args.test_data:
        raise ValueError('Provide at least one of --train-data or --test-data.')

    train_if_requested(args)

    if not args.test_data:
        return

    output_folder = args.output_folder
    if output_folder is None:
        raise ValueError('--output-folder is required when --test-data is provided.')

    predictions_file = run_inference(
        data_folder=args.test_data,
        model_folder=args.model_folder,
        output_folder=output_folder,
        verbose=args.verbose,
        allow_failures=args.allow_failures,
    )
    evaluate_if_possible(args.test_data, predictions_file, args.score_file, args.verbose)


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))