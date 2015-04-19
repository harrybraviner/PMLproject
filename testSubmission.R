testing <- read.csv('pml-testing.csv')
test <- testing[,cleanCols]
test <- test[,8:ncol(test)]
testPC <- predict(pp, test[,-53])
testPred <- predict(FM, newdata = testPC)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(testPred)
