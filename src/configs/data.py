import datetime
from .shared import data_raw


def get_beginning_of_month():
    # Get the current date
    current_date = datetime.datetime.now()

    # Get the first day of the current month
    first_day_of_month = current_date.replace(day=1).date()

    return first_day_of_month


# Data Download
metadata_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
offset = 0
limit = None
end_date = get_beginning_of_month()
yf_progress_bar = False
yf_repair = True
yf_rounding = True
metadata_file_path = data_raw / "meta_data/"

# Data Processing
window_size = 60
