class ClientException(Exception):
    def __init__(self, http_code, message, headers, error_code=None, error_data=None):
        self.http_code = http_code
        self.message = message
        self.headers = headers
        self.error_code = error_code
        self.error_data = error_data

    def __str__(self):
        if self.error_code:
            return f'Bybit Error ({self.error_code}): {self.error_data}'
        else:
            return f'HTTP Error ({self.http_code}): {self.message}'


class ServerException(Exception):
    def __init__(self, http_code, message):
        self.http_code = http_code
        self.message = message
