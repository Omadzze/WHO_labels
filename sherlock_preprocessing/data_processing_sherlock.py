import os
import shutil

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

class DataProcessingSherlock:

    def __init__(self):
        pass

    def load_table_with_labels(self, table_csv_path, gt_csv_path, table_name):
        """
        Load only .csv files that contains labels, if label is __none__ then it simply not loads it

        :param table_csv_path: Path to the .csv files
        :param gt_csv_path: Ground truth csv file path
        :param table_name: Name of the table
        :return:
        """

        ground_truth = pd.read_csv(gt_csv_path)

        # Filter to just the rows for this table, and drop the "__none__" rows
        gt_tbl = (
            ground_truth[ (ground_truth['table_name'] == table_name) & (ground_truth['label'] != "__none__") ]
            .sort_values('column_index')
        )

        # Make as list and zero based
        cols_to_keep = gt_tbl['column_index'].astype(int).tolist()
        labels = gt_tbl['label'].tolist()

        # load tables
        df = pd.read_csv(table_csv_path, header=0)

        # return columns with matching tables
        return df.iloc[:, cols_to_keep], labels

    def data_cleaning(self, data):
        """
        Clean the data in dataframes, removes hashes tha was broken during processing

        :param data: dataframe to clean
        :return: cleaned dataframe
        """

        for col in data.select_dtypes(include="object"):

            # make a bool mask of rows where the cell contains 'x000D'
            mask = data[col].str.contains("x000D", na=False)
            data.loc[mask, col] = np.nan

        return data

    def write_list_parquet(self, combined_data, data_path):
        def flatten(v):
            # only join if v is a list/tuple; otherwise assume it’s already the right string
            if isinstance(v, (list, tuple)):
                return ",".join(str(x) for x in v)
            else:
                return str(v)

        # 1) flatten your 'values' column
        value_strs = combined_data['values'].apply(flatten).tolist()
        values_array = pa.array(value_strs, type=pa.string())

        # 2) grab the DataFrame’s index and cast to int64
        #    (if your index isn’t integer, cast to string instead)
        idx = combined_data.index
        if pd.api.types.is_integer_dtype(idx):
            index_array = pa.array(idx.astype('int64').tolist(), type=pa.int64())
        else:
            index_array = pa.array(idx.astype('str').tolist(), type=pa.string())

        # 3) declare a schema *with* __index_level_0__ first
        schema = pa.schema([
            ('__index_level_0__', index_array.type),
            ('values',           pa.string())
        ])

        # 4) build & write the table
        table = pa.Table.from_arrays([index_array, values_array], schema=schema)
        pq.write_table(table, data_path)

    def flatten_and_save(self, df, labels, output_folder, table_name, lang_md_path, train_ratio, val_ratio, test_ratio, random_seed=42):

        """
        Preprocess the data for the Sherlock format dataset, it concatenates together values,
        creates parquet files and saves them. Moreover, it creates new file called lagnuage.parquet
        which consts of languages.

        :param df: tables with correct labels, not _none_
        :param labels: ground truth labels
        :param output_folder: where all files will be saved
        :param table_name: name of the table
        :param lang_md_path: path to the language file
        :return:
        """

        os.makedirs(output_folder, exist_ok=True)

        data_path   = os.path.join(output_folder, "data.parquet")
        labels_path = os.path.join(output_folder, "labels.parquet")
        lang_path   = os.path.join(output_folder, "language_md.parquet")

        # Load language
        lang_meta = pd.read_csv(lang_md_path)
        lang_row  = lang_meta.loc[lang_meta['table_name']==table_name, 'language']
        if lang_row.empty:
            raise ValueError(f"No language entry for {table_name}")
        language = lang_row.iloc[0]

        # Read existing parquet if it exists
        if os.path.exists(data_path):
            existing_data     = pd.read_parquet(data_path,   engine='fastparquet')
            existing_labels   = pd.read_parquet(labels_path, engine='fastparquet')
            existing_lang     = pd.read_parquet(lang_path,   engine='fastparquet')
            #next_idx = int(existing_data['index'].max()) + 1
        else:
            # Creates new parquet file
            existing_data     = pd.DataFrame(columns=['values'])
            existing_labels   = pd.DataFrame(columns=['type'])
            existing_lang     = pd.DataFrame(columns=['language'])
            #next_idx = 1

        # Build new rows
        value_rows = []
        label_rows = []
        lang_rows  = []
        #idx = next_idx
        for col_name, lbl in zip(df.columns, labels):
            # appends only unqiue elements, drop nons and makes flatten list
            uniq = df[col_name].dropna().unique().tolist()
            if not uniq:
                continue
            value_rows.append({'values': uniq,})
            label_rows.append({'type': lbl})
            lang_rows.append ({'language': language})
            #idx += 1

        # Creates temprorary dataframe
        new_data_df = pd.DataFrame(value_rows)
        new_label_df = pd.DataFrame(label_rows)
        new_lang_df = pd.DataFrame(lang_rows)

        # Concatenates old values with new values
        combined_data = pd.concat([existing_data,  new_data_df],  ignore_index=True)
        combined_labels = pd.concat([existing_labels,new_label_df],ignore_index=True)
        combined_lang = pd.concat([existing_lang,  new_lang_df],  ignore_index=True)

        # Saves everything as .parquet file
        #combined_data.to_parquet(data_path,   engine='pyarrow', index=False)
        self.write_list_parquet(combined_data, data_path)
        combined_labels.to_parquet(labels_path,engine='pyarrow', index=True)
        combined_lang.to_parquet(lang_path,   engine='pyarrow', index=True)

        print(f"Combined data length: {len(combined_data)}")
        print(f"Combined labels length: {len(combined_labels)}")
        print(f"Combined lang length: {len(combined_lang)}")

        return self.split_and_save(combined_data, combined_labels, combined_lang,
                                   output_folder, train_ratio, val_ratio, test_ratio, random_seed)

        #return combined_data, combined_labels, combined_lang



    def split_and_save(
            self,
            combined_data: pd.DataFrame,
            combined_labels: pd.DataFrame,
            combined_lang: pd.DataFrame,
            output_folder: str,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            random_state: int = None) -> dict:
        """
        Shuffle and split combined_data and combined_labels into train, validation, and test sets,
        then save them as parquet files in output_folder.

        Returns a dict with file paths for each split.
        """
        # Ensure output directory exists
        # Grab a random permutation of the *original* indices
        idx = combined_data.sample(frac=1, random_state=random_state).index
        # Apply that permutation uniformly, then reset to 0…N-1
        data_shuffled   = combined_data.loc[idx].reset_index(drop=True)
        labels_shuffled = combined_labels.loc[idx].reset_index(drop=True)
        lang_shuffled   = combined_lang.loc[idx].reset_index(drop=True)

        # compute sizes and split as before...
        n = len(data_shuffled)
        n_train = int(train_ratio * n)
        n_val   = int(val_ratio * n)

        # 3) Create splits
        # Data
        train_data = data_shuffled.iloc[:n_train]
        val_data   = data_shuffled.iloc[n_train:n_train + n_val]
        test_data  = data_shuffled.iloc[n_train + n_val:]

        # Labels
        train_labels = labels_shuffled.iloc[:n_train]
        val_labels   = labels_shuffled.iloc[n_train:n_train + n_val]
        test_labels  = labels_shuffled.iloc[n_train + n_val:]

        # Language
        train_language = lang_shuffled.iloc[:n_train]
        val_language   = lang_shuffled.iloc[n_train:n_train + n_val]
        test_language  = lang_shuffled.iloc[n_train + n_val:]

        # 4) Save splits
        # Train set
        train_data_path   = os.path.join(output_folder, 'train_data.parquet')
        train_labels_path = os.path.join(output_folder, 'train_labels.parquet')
        train_lang_path = os.path.join(output_folder, 'train_language.parquet')
        self.write_list_parquet(train_data, train_data_path)
        train_labels.to_parquet(train_labels_path, engine='pyarrow', index=True)
        train_language.to_parquet(train_lang_path, engine='pyarrow', index=True)

        # Validation set
        val_data_path   = os.path.join(output_folder, 'validation_data.parquet')
        val_labels_path = os.path.join(output_folder, 'validation_labels.parquet')
        val_lang_path = os.path.join(output_folder, 'val_language.parquet')
        self.write_list_parquet(val_data, val_data_path)
        val_labels.to_parquet(val_labels_path, engine='pyarrow', index=True)
        val_language.to_parquet(val_lang_path, engine='pyarrow', index=True)

        # Test set
        test_data_path   = os.path.join(output_folder, 'test_data.parquet')
        test_labels_path = os.path.join(output_folder, 'test_labels.parquet')
        test_lang_path = os.path.join(output_folder, 'test_language.parquet')
        self.write_list_parquet(test_data, test_data_path)
        test_labels.to_parquet(test_labels_path, engine='pyarrow', index=True)
        test_language.to_parquet(test_lang_path, engine='pyarrow', index=True)

        #print("Length")
        #print(f"Train data length: {len(train_data)}")
        #print(f"Train labels length: {len(train_labels)}")
        #print(f"Train language length: {len(train_language)}")

        return {
            'train':      (train_data_path, train_labels_path, train_lang_path),
            'validation': (val_data_path,   val_labels_path, val_lang_path),
            'test':       (test_data_path,  test_labels_path, test_lang_path)
        }
