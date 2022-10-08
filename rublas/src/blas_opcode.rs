// TODO move to CRT instructions to register
// impl From<CompleteStr<'_>> for OpCode {
//     fn from(s: CompleteStr<'_>) -> Self {
//         match s {
//             CompleteStr("crt.blas.addf") => BlasOpCode::ADDI32,
//             CompleteStr("crt.blas.addd") => BlasOpCode::SUBI32,
//             CompleteStr("crt.mul.i32") => OpCode::MULI32,
//             CompleteStr("crt.floordiv.i32") => OpCode::FLOORDIVI32,
//             CompleteStr("crt.literal.const.i32") => OpCode::CONSTI32,
//             CompleteStr("crt.literal.const.f32") => OpCode::CONSTF32,
//             CompleteStr("crt.literal.const.tensor") => OpCode::CONSTTENSOR,
//             CompleteStr("crt.add.f32") => OpCode::ADDF32,
//             CompleteStr("crt.sub.f32") => OpCode::SUBF32,
//             CompleteStr("crt.mul.f32") => OpCode::MULF32,
//             CompleteStr("crt.matmul.f32") => OpCode::MATMULF32,
//             CompleteStr("crt.div.f32") => OpCode::DIVF32,
//             _ => OpCode::ILLEGAL,
//         }
//     }
// }

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BlasOpCode {
    AddF,
    AddD,
    SubF,
    SubD,
    MulF,
    MulD,
    DivF,
    DivD,
    GemmF,
    GemmD,
    GemvF,
    GemvD,
}
