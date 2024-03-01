using Dates

function matlab_datenum_to_datetime(serial_date::Float64)
    # Compute the number of days since the MATLAB epoch (January 1, 0000)
    days_since_epoch = serial_date - 719529

    # Compute the time in seconds since the UNIX epoch (January 1, 1970)
    time_since_unix_epoch = days_since_epoch * 86400

    # Convert to DateTime object
    dt = DateTime(1970, 1, 1) + Second(time_since_unix_epoch)

    return dt
end

# Example usage
#matlab_serial_date = 737791.5  # Replace with your MATLAB serial date number
#julia_datetime = matlab_datenum_to_datetime(matlab_serial_date)
#println(julia_datetime)
