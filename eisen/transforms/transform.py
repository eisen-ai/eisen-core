class EisenBaseTransform:
    def __init__(self):
        self.expected_datatypes = []
        self.fields = []

    def _check_data_types(self, data):
        for field in self.fields:
            if len(self.expected_datatypes) > 0 and not isinstance(data[field], self.expected_datatypes):
                raise TypeError(
                    f'Data in field {field} of the data dictionary cannot be transformed. '
                    f'Its type ({type(data[field])}) does not match any of the allowed types '
                    f'{self.expected_datatypes}. Did you mean to use a different transform? '
                    f'Did you forget to include a transform? Are you trying to apply '
                    f'the transform to the wrong dictionary field?'
                )
