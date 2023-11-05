"""
This class takes care of loading the audio dataset for automatic speech recognition and preprocessing it.
The dataset is in the form of a .csv file with the following columns:
    -Gender: the gender of the speaker
    -Is_dysarthria: whether the speaker has dysarthria or not
    -Wav_path: the path to the .wav file
    -Txt_path: the path to the .txt file containing the prompt given to the speaker
    -Prompts: the prompt given to the speaker (same as in the .txt file)
We want to get a
"""

from datasets import load_dataset, Audio, Value, Features

class Dataset_Manager:

    def __init__(self):
        self.dataset = None

    def get_dataset(self):

        features = Features(
            {
                "gender": Value("string"),
                "audio": Audio(sampling_rate=16000),
                "text": Value("string"),
                "path": Value("string")
            }
        )

        # Note: to correctly load the dataset, either the python file must be run from inside the Customize_Dataset
        # folder or the paths within the .csv files must be changed to be relative to the Customize_Dataset folder
        sample_data =  load_dataset(
            'csv',
            data_files={
                'train': 'training_data_with_path_custom.csv',
                'test': 'testing_data_with_path_custom.csv',
                'valid': 'validation_data_with_path_custom.csv'
            }
        )

        sample_data = sample_data.cast(features)

        return sample_data





