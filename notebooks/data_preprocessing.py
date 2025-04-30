# Import statements
import pandas as pd
import numpy as np

# Read in data
faults = pd.read_csv("data/J1939Faults.csv")
diagnostics = pd.read_csv("data/VehicleDiagnosticOnboardData.csv").pivot(index='FaultId', columns='Name', values='Value') # Immediately pivot diagnostic data on FaultId to then be merged
faults_and_diagnostics = faults.merge(diagnostics, how='left', left_on='RecordID', right_on='FaultId')

def remove_service_locations(df, radius=0.05):
    """
    Remove data points that are near service locations.
    """
    # Define service locations
    service_locations = {
        'Location1': (36.0666667, -86.4347222), 
        'Location2': (35.5883333, -86.4438888), 
        'Location3': (36.1950, -83.174722) 
    }

    # Calculate distance to each service location and filter out points within a certain radius (e.g., 5 km)
    for loc, coords in service_locations.items():
        df['DistanceTo' + loc] = np.sqrt((df['Latitude'] - coords[0])**2 + (df['Longitude'] - coords[1])**2)
        df = df[df['DistanceTo' + loc] > radius]  # Filter out points within 5 km (radius)

    return df

def create_target_cols(df):
    """
    Create target columns for the dataset.
    """
    # Create target column for Full Derate
    df['FullDerate'] = (df['spn'] == 5246).astype('int8')
    # Order data by truck (EquipmentID) and time
    df = df.sort_values(['EquipmentID', 'EventTimeStamp'])
    # Ensure time is datetime
    df['EventTimeStamp'] = pd.to_datetime(df['EventTimeStamp'])

    # Create target column for Derate in next two hours
    df['NextDerateTime'] = df.where(df['FullDerate'] == 1)['EventTimeStamp']
    df['NextDerateTime'] = df.groupby('EquipmentID')['NextDerateTime'].transform('bfill')
    df['HoursUntilNextDerate'] = (df['NextDerateTime'] - df['EventTimeStamp']).dt.total_seconds() / 3600.0
    df['DerateInNextTwoHours'] = np.where(df['HoursUntilNextDerate'] <= 2, 1, 0)

    return df

# Convert diagnostic columns to appropriate dtypes
for col, dtype in {
    "AcceleratorPedal":"float16",
    "BarometricPressure":"float16",
    "CruiseControlActive":"bool",
    "CruiseControlSetSpeed":"float16",
    "DistanceLtd":"float16",
    "EngineCoolantTemperature":"float16",
    "EngineLoad":"float16",
    "EngineOilPressure":"float16",
    "EngineOilTemperature":"float16",
    "EngineRpm":"float16",
    "EngineTimeLtd":"float16",
    "FuelLevel":"float16",
    "FuelLtd":"float32",
    "FuelRate":"float16",
    "FuelTemperature":"float16",
    "IgnStatus":"bool",
    "IntakeManifoldTemperature":"float16",
    "ParkingBrake":"bool",
    "Speed":"float16",
    "SwitchedBatteryVoltage":"float16",
    "Throttle":"float16",
    "TurboBoostPressure":"float16",
    "eventDescription":"str",
    "EquipmentID":"str"
}.items():
    if dtype == 'bool':
        faults_and_diagnostics[col] = faults_and_diagnostics[col].astype('bool')
    else:
        faults_and_diagnostics[col] = pd.to_numeric(faults_and_diagnostics[col], errors='coerce').replace([np.inf, -np.inf], np.nan)

# throw out anything after a derate for a short amount of time

# throw out data nearby service locations

def ffill_nans(df): # alternatively try interpolation, moving averages, KNeighbors
    """
    Forward fill values for each EquipmentID group, resetting after each FullDerate == 1.
    """
    def fill_group(group):
        group = group.sort_values('EventTimeStamp')
        segment = group['FullDerate'].eq(1).cumsum() # Create segments based on FullDerate == 1
        return group.groupby(segment).ffill()

    return df.groupby('EquipmentID', group_keys=False).apply(fill_group)

# Separate training and testing data based on before and after 2019-01-01
faults_and_diagnostics_train = ffill_nans(create_target_cols(remove_service_locations(faults_and_diagnostics[faults_and_diagnostics['EventTimeStamp']<'2019-01-01'])))
faults_and_diagnostics_test = ffill_nans(create_target_cols(remove_service_locations(faults_and_diagnostics[(faults_and_diagnostics['EventTimeStamp']>='2019-01-01') & (faults_and_diagnostics['EventTimeStamp']<='2024-01-01')])))

def xy_train_test_split(feature_cols, target_col):
    X_train = faults_and_diagnostics_train[feature_cols]
    X_test = faults_and_diagnostics_test[feature_cols]
    y_train = faults_and_diagnostics_train[target_col]
    y_test = faults_and_diagnostics_test[target_col]

    # create train and test dataframes
    train_df = pd.concat([y_train, X_train], axis=1).rename(columns={target_col: 'target'})
    test_df = pd.concat([y_test, X_test], axis=1).rename(columns={target_col: 'target'})

    return train_df, test_df

def save_to_csv(train_df, test_df, file_name):
    """
    Save the train and test dataframes to CSV files.
    """
    train_file_path = f"data/{file_name}_train.csv"
    test_file_path = f"data/{file_name}_test.csv"
    
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)

    print(f"Train and test dataframes saved to {train_file_path} and {test_file_path}.")