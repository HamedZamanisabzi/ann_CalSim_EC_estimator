import pandas as pd

def calculate_monthly_average(csv_filename):  
    # Read the CSV file
    df = pd.read_csv(csv_filename, parse_dates=[0])

    # Check if the first column is a datetime
    if not isinstance(df.iloc[0, 0], pd.Timestamp):
        raise ValueError("The first column must contain date values")

    # Set the first column (date) as the index
    df.set_index(df.columns[0], inplace=True)

    # Resample data to monthly and calculate the mean
    monthly_avg = df.resample('M').mean()

    return monthly_avg

csv_filename = 'C:/Users/hzamanis/Documents/ann_calsim_main/smscg_output_on.csv'
monthly_average = calculate_monthly_average(csv_filename)
print(monthly_average)
monthly_average.to_csv('C:/Users/hzamanis/Documents/ann_calsim_main/monthly_ave_smscg_output_on.csv')
