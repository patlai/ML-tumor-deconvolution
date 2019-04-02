class Solution {
    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] sums = new int[m][n];
        //sums[m-1][n-1] = grid[m-1][n-1];
        
        for(int i = m-1; i >= 0; i--){
            for (int j = n-1; j >=0; j--){
                int rightSum = (j+1 >= n) ? -1 : sums[i][j+1];
                int downSum = (i+1 >= m) ? -1 : sums[i+1][j];
                
                sums[i][j] = grid[i][j];
                
                if (rightSum == -1 && downSum == -1){
                    continue;
                } else if (rightSum == -1){
                    sums[i][j] += downSum;
                } else if (downSum == -1){
                    sums[i][j] += rightSum;
                } else {
                    sums[i][j] += Math.min(rightSum, downSum);
                }
            }
        }
        
        return sums[0][0];
    }
}