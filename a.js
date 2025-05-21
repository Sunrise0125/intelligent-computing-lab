function get_primes(arr) {
    // FIXME:

    let rr = arr.filter(function (a) { 
       if (a === 1) 
        {
            return false;
        }
        let i;
        for (i = 2; i <= Math.floor(Math.sqrt(a)); i++) {
            if (a % i === 0) {
                return false;
            }
        }
        return true;
});
return rr;
}

// 测试:
let
    x,
    r,
    arr = [];
for (x = 1; x < 100; x++) {
    arr.push(x);
}
r = get_primes(arr);
if (r.toString() === [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97].toString()) {
    console.log('测试通过!');
} else {
    console.log('测试失败: ' + r.toString());
}
