#!/bin/bash

# Define the start and end dates
startDate="02/05/2021"
endDate="02/07/2021"

# Convert dates to a format suitable for looping
currentDate=$(date -d "$startDate" +%Y%m%d)
endDateFormatted=$(date -d "$endDate" +%Y%m%d)

# Loop through each day
while [[ "$currentDate" -le "$endDateFormatted" ]]; do
    # Format current date for output
    formattedStartDate=$(date -d "$currentDate" +"%m/%d/%Y")

    # The end date will be the same as the start date for daily intervals
    formattedEndDate=$formattedStartDate

    echo "Running globalDownload.py for dates: $formattedStartDate to $formattedEndDate"

    globalDownload.py --verbose \
                      --url=http://cedar.openmadrigal.org \
                      --outputDir=data/madrigal \
                      --user_fullname="Haig+Bananian" \
                      --user_email=hbananian@gmail.com \
                      --user_affiliation="APL" \
                      --format="hdf5" \
                      --startDate="$formattedStartDate" \
                      --endDate="$formattedEndDate" \
                      --inst=8308

    echo "Completed download for date: $formattedStartDate"

    # Move to the next day
    currentDate=$(date -d "$currentDate + 1 day" +%Y%m%d)
done
