(* Define the function to calculate PSL based on R *)
CalculatePSL[R_] := Module[
  {Q, epsilon, T, Rblock, I12, M, rowVec, colVec, PSL},
  
  (* Step 1: Normalize R to create Q *)
  epsilon = 10^-10;  (* A small value to avoid division by zero *)
  
  Q = Table[
    R[[i]] / (Total[R[[i]]] + epsilon), 
    {i, 1, 14}
  ];
  
  (* Step 2: Block Extraction of Q *)
  (* Extracting T (12x12 block for transient states) *)
  T = Q[[1 ;; 12, 1 ;; 12]];
  
  (* Extracting Rblock (12x2 block for transition from transient to absorbing states) *)
  Rblock = Q[[1 ;; 12, 13 ;; 14]];
  
  (* Step 3: Identity Matrix *)
  I12 = IdentityMatrix[12];
  
  (* Step 4: Matrix inversion and row vector *)
  M = Inverse[I12 - T];
  rowVec = Prepend[ConstantArray[0, 11], 1];
  colVec = {0, 1};
  
  (* Step 5: Compute PSL *)
  PSL = rowVec . M . Rblock . colVec;
  
  (* Return the symbolic PSL *)
  PSL
];

(* Example: Call the function with R defined earlier *)
R = Table[
  NormalDistribution[Symbol["mu" <> ToString[i] <> ToString[j]], Symbol["sigma" <> ToString[i] <> ToString[j]]],
  {i, 14}, {j, 14}
];

(* Compute PSL based on R *)
PSL = CalculatePSL[R];
PSL
