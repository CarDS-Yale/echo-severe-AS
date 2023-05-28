for cohort in '051823_full_test_2016-2020' '100122_test_2021' 'cedars'; do
    for eval_type in 'studies' 'videos' 'one_video'; do
        nohup python get_results.py --cohort "$cohort" --eval_type "$eval_type" --alpha 0.95 --n_samples 10000 &
    done
done