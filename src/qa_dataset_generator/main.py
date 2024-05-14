#!/usr/bin/env python
from qa_dataset_generator.crew import QaDatasetGeneratorCrew


def run():
    """
    A function to run the QaDatasetGeneratorCrew by kicking off with provided inputs.
    """
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'library_name': 'https://github.com/joaomdmoura/crewAI-tools/blob/main/crewai_tools/tools/website_search/README.md'
    }
    QaDatasetGeneratorCrew().crew().kickoff(inputs=inputs)