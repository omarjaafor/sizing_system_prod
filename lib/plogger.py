from google.cloud import logging
from datetime import datetime

LOGGER_DICT = {"LOGGER": None}


def create_logger(name):
    """
    Creates a google cloud logger `google.cloud.logging.logger.Logger`

        Parameters
        ----------
        name: str
            the name of the logger to be constructed.
    """
    client = logging.Client()
    LOGGER_DICT["LOGGER"] = client.logger(name)


def log(message, *, severity="DEBUG", labels=None, logger_name=None):
    """
    Log a text message.

        It may create a logger if logger_name is passed, and logger is not
        already created.

        For more information, see
        https://cloud.google.com/logging/docs/reference/v2/rest/v2/entries/write

        Parameters
        ----------
        message: str
            the log message.
        severity: Union['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'], optioanl, default: 'DEBUG'
            severity of event being logged.
        logger_name: str, optional, default: None
            The name of the logger to be constructed if not already.
        labels: dict, optional, default: None
            Mapping of labels for the entry.
            Labels that are added to the labels field of the log entry.

        Raises
        ------
        Exception
            if no logger is already created and logger_name is none
    """
    logger = LOGGER_DICT["LOGGER"]
    if logger is None and logger_name is None:
        raise ValueError("No logger exists, and logger_name is None")
    elif logger_name is not None:
        client = logging.Client()
        logger = client.logger(logger_name)

    logger.log_text(message, severity=severity, labels=labels)


def create_log_message(message, script=None, function=None, severity="DEBUG"):
    """
    Function to standardize log message

    Parameters
    ----------
    message: str,
        a log message
    script: str,
         the filename of executed script (__file__)
    function:
        the function where the logging happened __function__.name
    severity:
        the severity of the event to be logged
    Returns
    -------
    str
        a standard format message
    """
    now = datetime.now()
    script_name = script.split("/")[-1]
    log_message = "{time} {script}: {severity} {function}(): {message}".format(
        time=now,
        script=script_name,
        severity=severity,
        function=function,
        message=message,
    )
    return log_message
