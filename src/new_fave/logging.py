import hashlib
import datetime
import logging

fe_logger = logging.getLogger("fave-extract")
fe_logger.setLevel(logging.DEBUG)

def setup_logger() -> datetime.datetime:
    datestr = datetime.datetime.now()
    datehash = hashlib.sha1()
    datehash.update(str(datestr).encode("utf-8"))

    datehashtr = datehash.hexdigest()[:10]

    fe_handler = logging.FileHandler(
        filename=f"fave-extract-log-{datehashtr}.log"
    )

    fe_logger.addHandler(fe_handler)
    fe_logger.info(
        f"fave-extract started ({str(datestr)})"
    )
    return datestr