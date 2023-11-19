#!/bin/bash

VERBOSE=false
COMMAND=""
while getopts ":vc:" opt; do
  case ${opt} in
    v )
      VERBOSE=true
      ;;
    c )
      COMMAND=$OPTARG
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

echo -e "Running: ${1}"
echo -e "$1" | python
