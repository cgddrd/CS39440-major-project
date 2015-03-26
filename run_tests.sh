cd "src/"

for i in $(find . -type d -maxdepth 1)
do
  if [ $i != "." ]; then

    cd $i
    pwd

    if [ -f run_test.sh ]; then
      bash run_test.sh
    fi

    cd ..

  fi
done
