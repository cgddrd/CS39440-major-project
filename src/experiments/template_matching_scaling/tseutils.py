__author__ = 'connorgoddard'


class TSEUtils:

    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def get_smallest_key_dict(dict):
        return min(dict, key=dict.get)

    @staticmethod
    def get_smallest_key_value_dict(dict):
        smallest_dict_key = min(dict, key=dict.get)
        return dict[smallest_dict_key]

    @staticmethod
    def string_list_to_int_list(string_list):
        return map(TSEUtils.convert_to_int, string_list)

    @staticmethod
    def convert_to_int(value):
        return map(int, value)