import os
from glob import glob
import pandas as pd
import opensmile
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
import numpy as np


class EmoDB_data:
    """
    Class to handle the EmoDB data.
    
    Methods:
    ---------
    _get_file_codes(): list
        Returns a list of strings with codes for each audiofile that contain info on
        Actor, Emotion, etc.
    
    _interpret_file_codes():
    
    
    """
    def __init__(self):
        package_dir = os.path.dirname(os.path.abspath(__file__))
        self.folder = os.path.join(package_dir, "EmoDB/wav/")
        
        self._files = glob(f"{self.folder}*.wav")
        self._file_codes = self._get_file_codes()
        self.data = self._interpret_file_codes()
        self.speakers = self.get_speaker_info()
        self.sentences = self.get_sentences_info()
        self.emotions = self.get_emotions_info()
        
    def _get_file_codes(self) -> list:
        """
        Returns a list of the file_names without the .wav ending
        """
        file_codes = list()
        for file in self._files:
            f = file.replace(self.folder, "")
            f = f.replace(".wav", "")
            file_codes.append(f)
        return file_codes

    def _interpret_file_codes(self) -> pd.DataFrame:
        """
        Interprets the file's names according to Emo-DB website:

        Positions 1-2: number of speaker
        Positions 3-5: code for text
        Position 6: emotion (sorry, letter stands for german emotion word)
        Position 7: if there are more than two versions these are numbered a, b, c ....
        
        returns
        -------
            files_df: pd.DataFrame
            DataFrame with info on files, with columns "file_name", "code", "speaker", 
            "text_code", "emotion_code", "version"
        """
        
        files_df = pd.DataFrame()
        files_df["file_name"] = self._files
        files_df["code"] = self._file_codes
        
        files_df["speaker"] = [file_code[:2] for file_code in self._file_codes]
        files_df["text_code"] = [file_code[2:5] for file_code in self._file_codes]
        files_df["emotion_code"] = [file_code[5] for file_code in self._file_codes]
        files_df["version"] = [file_code[6] for file_code in self._file_codes]
        return files_df
        
    def get_speaker_info(self)-> pd.DataFrame:
        """
        Returns a DataFrame with info on the speakers:
            id (index), and columns "gender" and "age"
        """
        speaker_info = {
            "03": {"gender": "male", "age": "31"},
            "08": {"gender": "female", "age": "34"},
            "09": {"gender": "female", "age": "21"},
            "10": {"gender": "male", "age": "32"},
            "11": {"gender": "male", "age": "26"},
            "12": {"gender": "male", "age": "30"},
            "13": {"gender": "female", "age": "32"},
            "14": {"gender": "female", "age": "35"},
            "15": {"gender": "male", "age": "25"},
            "16": {"gender": "female", "age": "31"}
        }
        speakers_df = pd.DataFrame(speaker_info).transpose()
        return speakers_df
    
    def get_sentences_info(self)-> pd.DataFrame:
        """
        Returns the sentences in german and english based on their EmoDB code in the file names
        """
        
        sentences = {
            "a01": {"german": "Der Lappen liegt auf dem Eisschrank.",
                    "english": "The tablecloth is lying on the frigde."},
            "a02": {"german": "Das will sie am Mittwoch abgeben.",
                    "english": "She will hand it in on Wednesday."},
            "a04": {"german": "Heute abend könnte ich es ihm sagen.",
                    "english": "Tonight I could tell him."},
            "a05": {"german": "Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.",
                    "english": "The black sheet of paper is located up there besides the piece of timber."},
            "a07": {"german": "In sieben Stunden wird es soweit sein.",
                    "english": "In seven hours it will be."},
            "b01": {"german": "Was sind denn das für Tüten, die da unter dem Tisch stehen?",
                    "english": "What about the bags standing there under the table?"},
            "b02": {"german": "Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.",
                    "english": "They just carried it upstairs and now they are going down again."},
            "b03": {"german": "An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.",
                    "english": "Currently at the weekends I always went home and saw Agnes."},
            "b09": {"german": "Ich will das eben wegbringen und dann mit Karl was trinken gehen.",
                    "english": "I will just discard this and then go for a drink with Karl."},
            "b10": {"german": "Die wird auf dem Platz sein, wo wir sie immer hinlegen.",
                    "english": "It will be in the place where we always store it."}
        }
        sentences_df = pd.DataFrame(sentences).transpose()
        return sentences_df
    
    def get_emotions_info(self)-> pd.DataFrame:
        """
        Returns a emotional label meant by each letter code in the file names
        """
        emotions = {
            "W": {"emotion": "anger", "valence": "negative"},
            "L": {"emotion": "boredom", "valence": "negative"},
            "E": {"emotion": "disgust", "valence": "negative"},
            "A": {"emotion": "fear", "valence": "negative"},
            "F": {"emotion": "happiness", "valence": "positive"},
            "T": {"emotion": "sadness", "valence": "negative"},
            "N": {"emotion": "neutral", "valence": "positive"}
        }
        emotions_df = pd.DataFrame(emotions).transpose()
        return emotions_df

    def _get_audio_info(self, reset: bool =False) -> pd.DataFrame:
        """
        Returns a DataFrame with the opensmile features for each song in the dataset.
        Computes them again if they are not stored in "EmoDB/opensmile_data.csv"
        """
        package_dir = os.path.dirname(os.path.abspath(__file__))
        saved_file = os.path.join(package_dir, "EmoDB/opensmile_data.csv")

        if not reset and os.path.exists(saved_file):
            return pd.read_csv(saved_file)


        audio_df = pd.DataFrame()
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        for file_path in tqdm(self._files):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"'{file_path}' was not found")
            # if not file_path.endswith(".wav"):
            #     raise ValueError("File is not .wav")

            # gets loads of data about that piece of sound
            data = smile.process_file(file_path)

            data = pd.DataFrame(data).reset_index()
            audio_df = pd.concat([audio_df, data])

        audio_df.to_csv(saved_file, index=False)

        return audio_df


    def get_train_val_test_folds(self, k_folds: int=5, labeling: str="emotion"):        
        """
        Returns the necessary datasets for training and testing with the EmoDB dataset.
        Returns training and validation folds for the X features and y labels, as well as the combined
        trained and validation set (X_train_val, y_train_val) and the test sets.
        Train/Test separation and the K_folding of the datasets are based on grouping, not allowing
        samples by any speeker to be in more than one of the sets at the same time.
        This assures that voices that are being used in testing were not in the training set, 
        increasing validity of the empirical risk measurements.
        
        Params
        -------
        k_folds: int
            the number of folds in for the training and validation datasets
        labeling: str
            either "emotion" of "valence". With "emotion", the emotional labels returned are the
            original by EmoDB. With "valence", the y labels are transormed to either "positive" or "negative"
            valence values.
            
        
        Returns
        -------
        
        folds_data:
            list of dictionaries with keys "X_train", "X_val", "y_train", "y_val" for each k_fold, 
            where each of these are np.arrays with the training and validation X features and y labels
        X_train_val:
            np.array with training and validation features
        y_train_val:
            np.array with training and validation labels
        X_test:
            np.array with test features
        y_test:
            np.array with test labels
        """
        
        file_meta = self._interpret_file_codes()

        data = self._get_audio_info()
        data = data.drop(["start", "end"], axis=1).merge(file_meta[["file_name", "code"]],
                                                         how="left", left_on="file",
                                                         right_on="file_name")
        data = data.drop(["file", "file_name"], axis=1)
        data = data.set_index("code")

        file_meta = file_meta.set_index("code")

        file_meta = file_meta.merge(self.emotions, how="left", left_on="emotion_code", right_index=True)
        file_meta = file_meta.merge(self.get_speaker_info(), how="left", left_on="speaker", right_index=True)
        groups = file_meta["speaker"].to_numpy()

        
        splitter = GroupShuffleSplit(test_size=0.25, n_splits=2, random_state=42)
        split = splitter.split(data, groups=groups)
        
        train_inds, test_inds = next(split)
        
        data_train = data.iloc[train_inds]
        data_test = data.iloc[test_inds]
        
        groups_train = groups[train_inds]
        
        file_meta_train = file_meta.iloc[train_inds]
        file_meta_test = file_meta.iloc[test_inds]
        
        X_train_val = data_train.to_numpy()
        y_train_val = file_meta_train[labeling].to_numpy()
        
        X_test = data_test.to_numpy()
        y_test = file_meta_test[labeling].to_numpy()
        
        
        splits = []
        
        kf = GroupKFold(n_splits=k_folds)
        for train, val in kf.split(X=data_train, groups=groups_train):
            splits.append({"train": train, "val": val})

        X = data_train.to_numpy()
        y = file_meta_train[labeling].to_numpy()

        folds_data = []
        for fold in splits:
            X_train, X_val = X[fold["train"]], X[fold["val"]]
            y_train, y_val = y[fold["train"]], y[fold["val"]]

            folds_data.append({"X_train": X_train, "X_val": X_val, "y_train": y_train, "y_val": y_val})

        return folds_data, X_train_val, y_train_val, X_test, y_test

    def get_full_training_data(self, labeling="emotion"):
        """
        Returns the full data set for traing of the final model
        
        Params
        -------
        labeling: str
            either "emotion" of "valence". With "emotion", the emotional labels returned are the
            original by EmoDB. With "valence", the y labels are transormed to either "positive" or "negative"
            valence values.
        
        Returns
        -------
        
        X: np.array
            matrix with all the training features dataset. each line is a sample
        yl: np.array
            array witha a list of the target labels

        """
        
        file_meta = self._interpret_file_codes()

        data = self._get_audio_info()
        data = data.drop(["start", "end"], axis=1).merge(file_meta[["file_name", "code"]],
                                                         how="left", left_on="file",
                                                         right_on="file_name")
        data = data.drop(["file", "file_name"], axis=1)
        data = data.set_index("code")

        file_meta = file_meta.set_index("code")

        file_meta = file_meta.merge(self.emotions, how="left", left_on="emotion_code", right_index=True)
        file_meta = file_meta.merge(self.get_speaker_info(), how="left", left_on="speaker", right_index=True)

        X = data.to_numpy()
        y = file_meta[labeling].to_numpy()

        return X, y
