import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

class DataProcessingSherlock:
    def __init__(self):
        pass

    def load_table_with_labels(self, table_csv_path, gt_csv_path, table_name):

        ground_truth = pd.read_csv(gt_csv_path)

        gt_tbl = (
            ground_truth[
                (ground_truth['table_name'] == table_name) &
                (ground_truth['label'] != "__none__")
                ]
            .sort_values('column_index')
        )

        cols_to_keep = gt_tbl['column_index'].astype(int).tolist()
        labels = gt_tbl['label'].tolist()

        df = pd.read_csv(table_csv_path, header=0)
        return df.iloc[:, cols_to_keep], labels

    def data_cleaning(self, data):
        for col in data.select_dtypes(include="object"):
            mask = data[col].str.contains("x000D", na=False)
            data.loc[mask, col] = np.nan
        return data

    def write_list_parquet(self, combined_data, data_path):
        def flatten(v):
            if isinstance(v, (list, tuple)):
                return ",".join(str(x) for x in v)
            return str(v)

        value_strs = combined_data['values'].apply(flatten).tolist()
        values_array = pa.array(value_strs, type=pa.string())

        idx = combined_data.index
        if pd.api.types.is_integer_dtype(idx):
            index_array = pa.array(idx.astype('int64').tolist(), type=pa.int64())
        else:
            index_array = pa.array(idx.astype('str').tolist(), type=pa.string())

        schema = pa.schema([
            ('__index_level_0__', index_array.type),
            ('values',           pa.string())
        ])

        table = pa.Table.from_arrays([index_array, values_array], schema=schema)
        pq.write_table(table, data_path)

    def flatten_and_save(self, df, labels, output_folder, table_name):
        """
        Preprocess data for Sherlock format: concatenates values, creates parquet files for data and labels.
        :param df: DataFrame with non-__none__ columns
        :param labels: list of labels for each column
        :param output_folder: destination folder for parquet outputs
        :param table_name: name of the table (for logs)
        :return: (combined_data, combined_labels)
        """
        os.makedirs(output_folder, exist_ok=True)

        data_path   = os.path.join(output_folder, "synthetic_test_data.parquet")
        labels_path = os.path.join(output_folder, "test_synthetic_labels.parquet")

        # Read existing parquet if present
        if os.path.exists(data_path) and os.path.exists(labels_path):
            existing_data   = pd.read_parquet(data_path,   engine='fastparquet')
            existing_labels = pd.read_parquet(labels_path, engine='fastparquet')
        else:
            existing_data   = pd.DataFrame(columns=['values'])
            existing_labels = pd.DataFrame(columns=['type'])

        # Build new rows
        value_rows = []
        label_rows = []
        for col_name, lbl in zip(df.columns, labels):
            uniq = df[col_name].dropna().unique().tolist()
            if not uniq:
                continue
            value_rows.append({'values': uniq})
            label_rows.append({'type': lbl})

        new_data_df   = pd.DataFrame(value_rows)
        new_label_df  = pd.DataFrame(label_rows)

        # Concatenate old and new
        combined_data   = pd.concat([existing_data,  new_data_df],   ignore_index=True)
        combined_labels = pd.concat([existing_labels, new_label_df], ignore_index=True)

        # Save parquet files
        self.write_list_parquet(combined_data, data_path)
        combined_labels.to_parquet(labels_path, engine='pyarrow', index=True)

        print(f"Combined data length: {len(combined_data)}")
        print(f"Combined labels length: {len(combined_labels)}")

        return combined_data, combined_labels
