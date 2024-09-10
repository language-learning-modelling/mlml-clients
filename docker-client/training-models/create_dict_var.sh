declare -A dict

# Remove curly braces and quotes, and replace commas with spaces for easy iteration
# Declare an associative array

# Combine all arguments into a single JSON string
json="$*"

# Ensure the JSON string is properly quoted
json=$(printf '%s' "$json")

# Initialize an empty associative array to hold processed key-value pairs
declare -A dict

# Function to convert JSON object to a string representation for nested values
convert_json_to_string() {
  local json_value="$1"
  echo "$json_value" | jq -c .
}

# Process each key-value pair
for key in $(echo "$json" | jq -r 'keys[]'); do
    value=$(echo "$json" | jq -r --arg k "$key" '.[$k]')
    if echo "$value" | jq -e . >/dev/null 2>&1; then
        # Value is a JSON object, convert it to a string representation
        dict[$key]="object=>"$(convert_json_to_string "$value")
    else
        # Value is a simple value
        dict[$key]=$value
    fi
done

# Create a new JSON string from the associative array
new_json="{"
for key in "${!dict[@]}"; do
    # Escape double quotes in values
    value=$(echo "${dict[$key]}" | sed 's/"/\\"/g')
    new_json+="\"$key\":\"$value\","
done

# Print out the dictionary content
for key in "${!dict[@]}"; do
    echo "$key: ${dict[$key]}"
done
