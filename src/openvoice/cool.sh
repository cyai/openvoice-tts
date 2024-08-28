#!/bin/bash

# Output file
output_file="output.txt"

# List of files
files=(
  "__init__.py"
  "api.py"
  "attentions.py"
  "commons.py"
  "mel_processing.py"
  "models.py"
  "modules.py"
  "openvoice_app.py"
  "se_extractor.py"
  "text/__init__.py"
  "text/cleaners.py"
  "text/english.py"
  "text/mandarin.py"
  "text/symbols.py"
  "transforms.py"
  "utils.py"
)

# Clear the output file if it already exists
> $output_file

# Loop through the list of files
for file in "${files[@]}"
do
  # Add the file name as a heading
  echo "## $file" >> $output_file
  echo "" >> $output_file
  echo "<code>" >> $output_file
  echo "" >> $output_file
  
  # Append the contents of the file
  cat "$file" >> $output_file
  
  echo "" >> $output_file
  echo "</code>" >> $output_file
  echo "" >> $output_file
done

echo "Files have been copied into $output_file"
