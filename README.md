# IA_evoman
Projeto da disciplina de Inteligência Artificial UFABC

## Arquivos
Nossos arquivos são os [our_controller.py](our_controller.py) e [our_optimization.py](our_optimization.py).  
Os resultados são gravados na pasta `our_tests`, sendo eles:
* [evoman_logs.txt](our_tests/evoman_logs.txt): os logs do evoman
* [results.txt](our_tests/results.txt): log das gerações e melhor fitness da população
* [Evoman.pkl](our_tests/Evoman.pkl): pesos das redes da população

## Preparando o ambiente
Verifique se o python3 está presente em seu computador através do comando `python3 --version`

Caso não esteja, instale-o:  

**Ubuntu/Debian:**  
`sudo apt-get install python3`  
`sudo apt-get install python3-pip`

Instale também as bibliotecas utilizadas:  
`pip3 install Keras`  
`pip3 install numpy`  
`pip3 install tensorflow`  
`pip3 install pygame`  

## Executando

Para executar o programa, utilize o comando `python3 our_optimization.py`.  

É possível alternar entre o modo de treino e teste, mudando o valor da flag `mode` no arquivo `our_optimization.py` para _'train'_ ou _'test'_. Também é possível definir os inimigos que serão utilizados para o treino, por meio da variável `enemies`, no mesmo arquivo.

No mode de treino, a interface gráfica é desativada e, ao final das iterações definidas ao instânciar a classe _GA_, é feito um teste com todos os bosses.

## Resultados e Conclusões
Elaboramos um relatório com nossos resultados e conclusões, disponível em [Evoman-Relatorio.pdf](Evoman-Relatorio.pdf)  
