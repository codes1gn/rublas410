
#[cfg(test)]
mod tests {
    use ndarray::arr2;

    #[test]
    fn it_works() {
        let a = arr2(&[[1, 2, 3],
                       [4, 5, 6]]);

        let b = arr2(&[[6, 3],
                       [5, 2],
                       [4, 1]]);

        println!("{}", a.dot(&b));

        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
