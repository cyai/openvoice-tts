
#!/bin/bash

# Output file
output_file="output.txt"

# List of files
files=(
  "__init__.py"
  "core/__init__.py"
  "core/libs/__init__.py"
  "core/libs/model.py"
  "core/schemas/__init__.py"
  "core/schemas/synthesize_schema.py"
  "core/settings.py"
  "main.py"
  "v1/__init__.py"
  "v1/api.py"
  "v1/endpoints/__init__.py"
  "v1/endpoints/synthesize.py"
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

