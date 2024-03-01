using Dates

# Example usage
#date_example = Date(2024, 2, 29)
#datetime_example = DateTime(2024, 2, 29, 12, 30, 0)
#println("Julian day number for $date_example: ", julian_day_number(date_example))
#println("Julian date for $datetime_example: ", julian_date(datetime_example))
# Define a function to calculate Julian day number

#function julian_day_number(date::Date)
function julian_date(date::Date)
    # Calculate the Julian day number
    jd = 2415020 + 365 * (year(date) - 1900) + Dates.dayofyear(date)
    return jd
end

# Define a function to calculate Julian date
function julian_date(date::DateTime)
    # Calculate the Julian day number
    jd = 2415020 + 365 * (year(date) - 1900) + Dates.dayofyear(date)    
    # Add the fractional part of the day
    jd += (Dates.hour(date) * 3600 + Dates.minute(date) * 60 + Dates.second(date)) / 86400
    
    return jd
end

