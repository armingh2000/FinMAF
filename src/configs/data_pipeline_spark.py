import datetime


def get_beginning_of_month():
    # Get the current date
    current_date = datetime.datetime.now()

    # Get the first day of the current month
    first_day_of_month = current_date.replace(day=1).date()

    return first_day_of_month


# configs for data_pipeline/spark/
offset = 0
limit = 20
# Until the first day of the current month
end_date = get_beginning_of_month()
yfinance_progress_bar = False
