import unittest
import pandas as pd

import sys
sys.path.insert(0, '../preprocess')

import condense_csv

class Test_condense_csv(unittest.TestCase):
    def test_downcast_df_int_columns(self):
        """
        Test that it can downcast integer columns.
        """

        data = pd.DataFrame({
            'Float64': [1.0],
            'Int64': [1],
            "Object": 'Hi'
        })

        result = condense_csv.downcast_df_int_columns(data)
        result = [str (i) for i in list (result.dtypes.values)]
        answer = ['float64', 'int8', 'object']

        self.assertEqual(result, answer)

    def test_convert_obj_columns_to_cat(self):
        """
        Test that it can convert objects to categories
        """

        data = pd.DataFrame({
            'Float64': [1.0],
            'Int64': [1],
            "Object": 'Hi'
        })

        result = condense_csv.convert_obj_columns_to_cat(data, {'wpt_name'})
        result = [str (i) for i in list (result.dtypes.values)]
        answer = ['float64', 'int64', 'category']

        self.assertEqual(result, answer)

    def test_compress_X(self):
        """
        Test that it can convert objects to categories
        """

        data = pd.DataFrame({
            'date_recorded': ['2011-03-14'],
            'Int64': [1],
            "Object": ['Hi']
        })

        result = condense_csv.compress_X(data)
        result = [str (i) for i in list (result.dtypes.values)]
        answer = ['int8', 'category', 'datetime64[ns]']

        self.assertEqual(result, answer)

if __name__ == '__main__':
    unittest.main()