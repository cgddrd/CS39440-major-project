set -e

# Go into the code source directory.
cd "src/"

# Loop through the first level of directories inside the 'src' directory.
for i in $(find . -type d -maxdepth 1)
do

  # We need to exclude the current directory (i.e. '.') returned by 'find'
  if [ $i != "." ]; then

    # Move into each project folder in turn.
    cd $i

    echo
    echo "Entering project: $i"
    echo

    # Check to see if the project folder contains a build/test script, if so run.
    if [ -f run_travisci_proj_test.sh ]; then
      echo
      echo "Travis CI build/test script found. Running.."
      echo
      bash run_travisci_proj_test.sh
    else
      echo
      echo "Project does not contain Travis CI build/test script. Skipping.."
      echo
    fi

    # Once finished, move back into the parent directory ('src') and move along.
    echo
    echo "Moving up to parent folder."
    echo
    cd ..

  fi
done
