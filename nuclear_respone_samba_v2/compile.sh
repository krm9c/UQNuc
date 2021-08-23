#!/bin/bash -x
# Nuclear_Respone_Inv

if [ "${1}" == "compile" ] ; then
   python Samba_test_new.py compile --pef-name="inverse.pef" --output-folder=./out --debug
elif [ "${1}" == "mp" ] ; then
   python Samba_test_new.py measure-performance  --pef-name="inverse.pef" --output-folder=./out --debug
elif [ "${1}" == "pdb" ] ; then
   python -m pdb Samba_test_new.py compile --pef-name="inverse.pef" --output-folder=./out --debug
elif [ "${1}" == "inference" ] ; then
   python Samba_test_new.py compile --inference --pef-name="inverse" --output-folder=./out --debug
elif [ "${1}" == "mcpu" ] ; then
   python Samba_test_new.py measure-cpu --debug
elif [ "${1}" == "test" ] ; then
   python Samba_test_new.py test --pef="out/inverse/inverse.pef" --debug
elif [ "${1}" == "run" ] ; then
  python Samba_test_new.py run --pef="out/inverse/inverse.pef" --debug
fi
