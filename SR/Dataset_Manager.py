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


        sample_data =  load_dataset(
            'csv',
            data_files={
                'train': './Customized_Dataset/training_data_with_path_custom.csv',
                'test': './Customized_Dataset/testing_data_with_path_custom.csv',
                'valid': './Customized_Dataset/validation_data_with_path_custom.csv'
            }
        )

        #sample_data = sample_data.remove_columns(["__index_level_0__"])

        sample_data = sample_data.cast(features)

        return sample_data





