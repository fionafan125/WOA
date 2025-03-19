儲存excel 
confusion_storage:
    固定資料夾裡面的各個penalty的excel
    一定不會賺錢的我們直接設成-10 演算法不會執行

儲存演算法的收斂狀況及運算結果
data_file_penalty_1e3至1e6:
    懲罰函數的相乘的權重為:如標題
    machine amount_[number1]:
        year_[number2]:
            儲存各個配置的收斂結果，沒有的就是不可能會賺錢的就不跑演算法了
    
固定machine 數量畫摺線圖
machine_png:
    固定懲罰為[number1] 得到的數值
    penalty_[number1]:
        固定machine數量為[number2] 對year做實驗及畫出實驗圖
        machine_[number2].png

    
固定year 數量畫摺線圖
year_png:
    固定懲罰為[number1] 得到的數值
    penalty_[number1]:
        固定year數量為[number2] 對year做實驗及畫出實驗圖
        year_[number2].png