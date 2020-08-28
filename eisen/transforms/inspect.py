from typing import List


class GetShape:
    """
    Transform used to inspect the data in the data dictionary. This transform prints the shape of the data contained
    in the specified fields of the data dictionary.

    .. code-block:: python

        from eisen.transforms import GetShape
        tform = GetShape(['image', 'label'])
        tform = tform(data)

    Prints the shape of the numpy arrays stored in the data dictionary.
    """

    def __init__(self, fields: List[str]):
        """
        :param fields: A list of data dictionary field names which need to be investigated (print shape)
        :type fields: list of str

        .. code-block:: python

            from eisen.transforms import GetShape
            tform = GetShape(fields=['image', 'label'])
            tform = tform(data)

        """
        self.fields = fields

    def __call__(self, data: dict) -> dict:
        for field in self.fields:
            print("Data stored in data dictionary at field {} has shape {}".format(field, data[field].shape))

        return data


class GetContent:
    """
    Transform used to inspect the data in the data dictionary. This transform prints the content of the data contained
    in the specified fields of the data dictionary.

    .. code-block:: python

        from eisen.transforms import GetContent
        tform = GetContent(['image', 'label'])
        tform = tform(data)

    Prints the content of the data stored in the data dictionary.
    """

    def __init__(self, fields):
        """
        :param fields: A list of data dictionary field names which need to be investigated (print content)
        :type fields: list of str

        .. code-block:: python

            from eisen.transforms import GetContent
            tform = GetContent(fields=['image', 'label'])
            tform = tform(data)

        """
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            print(f"Data stored in data dictionary at field {field} {data[field]}")

        return data
