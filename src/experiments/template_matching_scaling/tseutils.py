__author__ = 'connorgoddard'


class TSEUtils:

    def __init__(self):
        # 'pass' is used when a statement is required syntactically but you do not want any command or code to execute.
        pass

    @staticmethod
    def string_list_to_int_list(string_list):
        return map(TSEUtils.convert_to_int, string_list)

    @staticmethod
    def convert_to_int(value):
        return map(int, value)