import datetime
from .shared import dps_raw


def get_beginning_of_month():
    # Get the current date
    current_date = datetime.datetime.now()

    # Get the first day of the current month
    first_day_of_month = current_date.replace(day=1).date()

    return first_day_of_month


# configs for data_pipeline/spark/
offset = 0
limit = 100
# Until the first day of the current month
end_date = get_beginning_of_month()
yf_progress_bar = False
yf_repair = True
yf_rounding = True
meta_file_path = dps_raw / "meta_data/"
