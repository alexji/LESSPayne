"""Functions to help with compatibility in renamed module from LESSPayne to lesspayne"""

import io
import pickle


class RenameUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        renamed_module = module.replace("LESSPayne", "lesspayne")
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj, **kwargs):
    return RenameUnpickler(file_obj, **kwargs).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)
