import sys

class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a")
        
    def __del__(self):
        self.log.close()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.log.flush()
        
    def __enter__(self):
        self.stdout_saved = sys.stdout
        sys.stdout = self

    def __exit__(self, exception_type, exception_value, traceback):
        sys.stdout = self.stdout_saved
        self.log.flush()
        self.log.close()
        
class LoggerStdErr(object):
    def __init__(self, path):
        self.terminal = sys.stderr
        self.log = open(path, "a")
        
    def __del__(self):
        self.log.close()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.log.flush()
        
    def __enter__(self):
        self.stderr_saved = sys.stderr
        sys.stderr = self

    def __exit__(self, exception_type, exception_value, traceback):
        sys.stderr = self.stderr_saved
        self.log.flush()
        self.log.close()