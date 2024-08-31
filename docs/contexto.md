# Introdução

<!-- Texto descritivo introdutório apresentando a visão geral do projeto a ser desenvolvido considerando o contexto em que ele se insere, os objetivos gerais, a justificativa e o público-alvo do projeto. -->

A saúde mental é um componente essencial para o bem-estar do ser humano, influenciando profundamente aspectos físicos, sociais e emocionais da vida. 

No cenário global atual, a prevalência de transtornos mentais tem se tornado uma questão alarmante, especialmente após os impactos da pandemia de COVID-19. Segundo a Organização Mundial da Saúde (OMS), no Brasil, estima-se que 9,3% da população adulta enfrente desafios relacionados à saúde mental, como ansiedade e depressão.

Diante deste panorama, há uma crescente demanda por alternativas aos tratamentos convencionais, que muitas vezes se concentram em abordagens farmacológicas. A busca por soluções complementares, que sejam acessíveis e minimamente invasivas, é uma prioridade para profissionais de saúde e pesquisadores. 

Nesse contexto, a música tem emergido como uma ferramenta terapêutica promissora. A musicoterapia, especificamente, tem sido objeto de estudos que demonstram sua eficácia na redução de sintomas de transtornos como ansiedade, depressão, insônia e Transtorno Obsessivo-Compulsivo (TOC).

O presente projeto Música e Saúde Mental, tem como objetivo investigar a relação entre o consumo de música e indicadores de saúde mental, utilizando um dataset que integra informações de usuários de aplicativos de música com suas autoavaliações sobre o impacto da música em seu bem-estar.

A escolha do tema é fundamentada na necessidade de explorar intervenções que possam complementar os tratamentos tradicionais, oferecendo uma opção de baixo custo e fácil acesso para a mitigação dos sintomas de transtornos mentais. Além disso, a música, por ser amplamente acessível e possui o potencial de alcançar um vasto público, tornando-se uma ferramenta poderosa na promoção da saúde mental.


## Problema

A música tem ajudado bastante na promoção do bem-estar emocional e mental no dia-a-dia, especialmente em tempos de grande estresse, como ocorreu durante a pandemia de COVID-19, por exemplo, o que trouxe uma restrição social e a ansiedade para o cotidiano, muitas pessoas encontraram na música um refúgio, conforto, assim como também conseguiram aliviar o peso emocional que a incerteza do período trouxe.

Além de proporcionar alívio imediato, a música tem um papel importante na redução dos sintomas de ansiedade, depressão, insônia e transtorno obsessivo-compulsivo (TOC). O quanto alguém escuta música e os tipos de gêneros musicais que prefere podem influenciar positivamente a saúde mental. Pesquisas recentes mostram que a exposição regular a determinados tipos de música pode diminuir de forma significativa os níveis de ansiedade e depressão, induzindo um estado de relaxamento e bem-estar (Aalbers et al., 2017).

Portanto, trazer a música para a rotina diária pode ser uma maneira acessível e eficaz de combater esses transtornos. Sendo assim, este trabalho busca investigar como a música utilizada pelo usuário pode contribuir para sua saúde mental, considerando informações sobre a frequência com que ele escuta música e seus gêneros preferidos. Com isso, esperamos que ao introduzir novos dados de usuários, utilizando o histórico do dataset possamos oferecer uma abordagem que possa prever o impacto futuro em sua saúde mental.

## Questão de pesquisa

Dado todos esses pontos em como a música pode vem sendo benéfica nos ultimos anos e como ela pode afetar na saúde mental, surge a questão do rumo que esta pesquisa tomará e o que iremos apresentar frente a todos os dados presentes no dataset. A pergunta em questão é: 
<br/><br/>"O quão efetivo cada genero musical é no tratamento dos sintomas de doenças mentais?"<br/><br/> 
Respondendo essa pergunta, um modelo conseguirá receber, analisar e sugerir genêros no qual as pessoas, que participaram do dataset, sentiram que a música melhorou a saúde mental.
## Objetivos preliminares  

### Objetivo Geral:
Explorar e experimentar modelos de aprendizado de máquina para analisar a relação entre o consumo de diferentes gêneros musicais e os indicadores de saúde mental, com o intuito de contribuir para uma aplicação mais eficaz da musicoterapia.

### Objetivos Específicos:
Predizer o impacto da frequência de escuta de diferentes gêneros musicais nos indicadores de saúde mental, como níveis de ansiedade, depressão e transtorno obsessivo-compulsivo (TOC), utilizando técnicas de aprendizado de máquina.

Investigar como cada estilo musical influencia os aspectos específicos da saúde mental e identificar padrões que possam ser utilizados para personalizar intervenções em musicoterapia.

Desenvolver um modelo preditivo que, a partir do consumo musical de um indivíduo, possa prever o impacto futuro em sua saúde mental, oferecendo insights para prevenir possíveis agravamentos de condições psicológicas.

## Justificativa

<!-- Descreva a importância ou a motivação para trabalhar com o conjunto de dados escolhido. Indique as razões pelas quais você escolheu seus objetivos específicos, as razões para aprofundar o estudo do problema identificado e qual o impacto que tal problema provoca na sociedade. Lembre-se de quantificar (com dados reais e suas respectivas fontes) este impacto.

> **Links Úteis**:
> - [Como montar a justificativa](https://guiadamonografia.com.br/como-montar-justificativa-do-tcc/) -->

A saúde mental é um dos pilares essenciais para a qualidade de vida, influenciando diretamente o bem-estar físico, social e emocional das pessoas. Estudos recentes indicam que uma parcela significativa da população mundial sofre de algum tipo de transtorno mental. De acordo com a Organização Mundial da Saúde (OMS, 2022), cerca de 1 em cada 8 pessoas no mundo vive com um transtorno mental, número que se agravou significativamente após a pandemia de COVID-19, que gerou um aumento de 25% na prevalência de ansiedade e depressão em nível global. No Brasil, estima-se que aproximadamente 9,3% da população adulta tenha sido diagnosticada com algum transtorno mental comum, como ansiedade ou depressão, conforme dados do Ministério da Saúde.

Diante desse cenário alarmante, torna-se imperativo buscar alternativas complementares aos tratamentos tradicionais, muitas vezes centrados no uso de medicamentos. Uma dessas alternativas é a música, cuja influência positiva na saúde mental tem sido amplamente discutida. A musicoterapia, por exemplo, tem demonstrado eficácia na redução dos sintomas de ansiedade, depressão, insônia e Transtorno Obsessivo-Compulsivo (TOC).

O presente projeto de dados visa explorar a relação entre o consumo de música através de aplicativos e a melhora nos sintomas de doenças mentais, com base em uma base de dados que combina informações de usuários e sua auto avaliação sobre os efeitos da música. A escolha desse tema se fundamenta na crescente necessidade de encontrar soluções acessíveis e não invasivas que possam complementar os tratamentos convencionais, oferecendo alívio e melhor qualidade de vida para aqueles que sofrem de transtornos mentais.

O impacto social desse estudo é potencialmente significativo, considerando que o acesso à música é amplamente democratizado e pode ser facilmente integrado ao cotidiano das pessoas. Analisando a eficácia da música como ferramenta terapêutica, pode-se promover a sua utilização como um meio acessível e eficaz para a mitigação dos sintomas de transtornos mentais, contribuindo assim para a saúde pública e para a redução do estigma associado a esses transtornos.

A relevância deste projeto, portanto, reside na possibilidade de oferecer uma abordagem inovadora e de baixo custo para o tratamento de condições de saúde mental, ao mesmo tempo em que se fortalece a compreensão do papel da música no bem-estar humano.

## Público-Alvo

<!-- Descreva quem serão as pessoas que poderão se beneficiar com a sua investigação indicando os diferentes perfis. O objetivo aqui não é definir quem serão os clientes ou quais serão os papéis dos usuários na aplicação. A ideia é, dentro do possível, conhecer um pouco mais sobre o perfil dos usuários: conhecimentos prévios, relação com a tecnologia, relações hierárquicas, etc.

Adicione informações sobre o público-alvo por meio de uma descrição textual, diagramas de personas e mapa de stakeholders. -->

O estudo beneficiará pessoas de várias idades, desde adolescentes até idosos, que têm diferentes graus de familiaridade com tecnologia e diferentes motivações para o uso de música em suas vidas. O projeto pode fornecer insights valiosos para profissionais de saúde mental sobre como personalizar intervenções musicais de acordo com o perfil etário e tecnológico dos pacientes. Além disso, pode orientar desenvolvedores de aplicativos de música a criar funcionalidades que melhor atendam às necessidades de seus usuários, promovendo bem-estar e saúde mental.

* Embora a base de dados utilizada possua avaliação de usuários de todas as idades, mais de 70% da base se concentra na faixa etária entre 14 e 27 anos.

> **Links Úteis**:
> - [Público-alvo](https://blog.hotmart.com/pt-br/publico-alvo/)
> - [Como definir o público alvo](https://exame.com/pme/5-dicas-essenciais-para-definir-o-publico-alvo-do-seu-negocio/)
> - [Público-alvo: o que é, tipos, como definir seu público e exemplos](https://klickpages.com.br/blog/publico-alvo-o-que-e/)
> - [Qual a diferença entre público-alvo e persona?](https://rockcontent.com/blog/diferenca-publico-alvo-e-persona/)

## Estado da arte

Nesta seção, deverão ser descritas outras abordagens identificadas na literatura que foram utilizadas para resolver problemas similares ao problema em questão. Para isso, faça uma pesquisa detalhada e identifique, no mínimo, 5 trabalhos que tenham utilizado dados em contexto similares e então: (a) detalhe e contextualize o problema, (b) descreva as principais características do _dataset_ utilizado, (c) detalhe quais abordagens/algoritmos foram utilizados (e seus parâmetros), (d) identifique as métricas de avaliação empregadas, e (e) fale sobre os resultados obtidos. 

> **Links Úteis**:
> - [Google Scholar](https://scholar.google.com/)
> - [IEEE Xplore](https://ieeexplore.ieee.org/Xplore/home.jsp)
> - [Science Direct](https://www.sciencedirect.com/)
> - [ACM Digital Library](https://dl.acm.org/)

A investigação sobre o impacto da música na saúde mental tem se mostrado uma área promissora, especialmente como uma alternativa terapêutica complementar às abordagens tradicionais. O uso da música, particularmente durante períodos de estresse e isolamento social, como os vividos durante a pandemia de COVID-19, tem se destacado por proporcionar benefícios significativos ao bem-estar emocional e mental. Com o aumento do interesse por terapias não invasivas e de fácil acesso, a música emergiu como um meio eficaz de intervenção.

Este estado da arte tem como objetivo revisar e analisar as abordagens mais relevantes presentes na literatura recente, focando em cinco estudos que exploram a relação entre o consumo de diferentes gêneros musicais e a saúde mental.

Essa revisão permitirá não apenas compreender as metodologias existentes, mas também identificar lacunas e oportunidades para futuras pesquisas, especialmente no desenvolvimento de modelos preditivos que possam ser aplicados na musicoterapia, visando melhorar a saúde mental de forma personalizada e eficaz.

Analisando o artigo "Music Classification and Mental Health Analysis using Exploratory Data Analysis" temos os seguintes resultados:
1. Contextualização do Problema <br/>
O impacto da música na saúde mental tem sido amplamente explorado como uma alternativa terapêutica, especialmente em tempos de crise, como a pandemia de COVID-19, onde a música serviu como um refúgio emocional. No estudo de Bhavani et al., o objetivo principal é explorar a relação entre diferentes gêneros musicais e condições de saúde mental, como ansiedade, depressão, insônia e Transtorno Obsessivo-Compulsivo (TOC)​(referenciar o artigo de acordo com a ABNT). O estudo utiliza a Análise Exploratória de Dados (EDA) e algoritmos de classificação para prever como os diferentes tipos de música podem influenciar o bem-estar mental dos indivíduos.

2. Revisão de Trabalhos Relacionados <br/>
**2.1 Características dos Datasets Utilizados** <br/>
2.1.1. Estudo de Bhavani et al. (2023): <br/>

* Dataset: Inclui 736 registros e 33 colunas, com dados coletados via formulários online distribuídos em várias plataformas sociais. As variáveis incluem dados demográficos, preferências musicais e autoavaliações de condições de saúde mental como ansiedade e depressão​​(referenciar o artigo de acordo com a ABNT).
* Pré-processamento: Limpeza dos dados para remover valores nulos e outliers, utilizando média e moda para imputação.

2.1.2. Markov e Matsui (2014): <br/>

* Dataset: Dois datasets de tamanho comparável, utilizados para classificação de gêneros musicais e estimativa de emoções. A análise comparou o desempenho dos algoritmos de Support Vector Machine (SVM) e Gaussian Process (GP), sendo o GP mais eficaz em termos de redução de erros e aumento da precisão​​(referenciar o artigo de acordo com a ABNT).

2.1.3. Xu et al. (2021): <br/>

* Dataset: Dados coletados de experimentos psicológicos, envolvendo características de personalidade e preferências por música triste. O estudo utiliza técnicas de machine learning para prever preferências musicais com base em gênero e traços de personalidade​​(referenciar o artigo de acordo com a ABNT).

**2.2 Abordagens/Algoritmos Utilizados** <br/>
2.2.1 Gaussian Naive Bayes (GNB): <br/>

Utilizado no estudo de Bhavani et al., o GNB é adequado para dados contínuos com distribuição Gaussiana. No contexto do estudo, o GNB foi empregado para classificar o efeito da música na saúde mental, resultando em uma acurácia de 68%​​(referenciar o artigo de acordo com a ABNT).

2.2.2. Support Vector Machine (SVM) e Gaussian Process (GP): 

No estudo de Markov e Matsui, ambos os algoritmos foram usados para tarefas de classificação de gênero musical e estimativa de emoções. O GP demonstrou melhor desempenho em comparação ao SVM, destacando sua eficácia em reduzir erros de classificação e melhorar a precisão na estimativa de emoções​​(referenciar o artigo de acordo com a ABNT).

2.2.3. Análise de Regressão e Classificação (SVM, Naive Bayes): 

No estudo de Xu et al., foram utilizadas abordagens de regressão e classificação para prever a preferência por música triste. A análise revelou interações significativas entre características de personalidade e preferências musicais, sendo o SVM um dos métodos mais eficazes para essa tarefa​​(referenciar o artigo de acordo com a ABNT).

2.3.1 Métricas de Avaliação 
**Acurácia**: Utilizada como métrica principal nos estudos de Bhavani et al. e Markov e Matsui, onde o desempenho dos modelos é avaliado pela porcentagem de previsões corretas. A acurácia varia entre 68% (GNB) a 92.3% (SVM) em diferentes contextos de análise​​(referenciar o artigo de acordo com a ABNT).

**Coeficiente de Determinação**: Usado para avaliar a precisão das previsões de emoções musicais no estudo de Markov e Matsui​​(referenciar o artigo de acordo com a ABNT).

2.4.1 Resultados Obtidos
Bhavani et al.: O estudo concluiu que diferentes gêneros musicais têm impactos variados na saúde mental, com ouvintes de rock, hip-hop e lofi music apresentando maiores níveis de depressão. A acurácia do modelo GNB foi de 68%, enquanto no Rapid Miner a acurácia atingiu 74%​(referenciar o artigo de acordo com a ABNT).

Markov e Matsui: O Gaussian Process superou o SVM em ambas as tarefas de classificação e estimação de emoções, demonstrando uma redução relativa de 13.6% no erro de classificação de gênero musical​(referenciar o artigo de acordo com a ABNT).

Xu et al.: O estudo identificou que homens são mais propensos a preferir música triste, e que as características de extroversão e gênero possuem interações significativas na preferência musical. Essas descobertas podem ser aplicadas para personalizar intervenções musicais em terapias​​(referenciar o artigo de acordo com a ABNT).

3. Discussão Comparativa
Os estudos revisados demonstram que a música tem um impacto significativo na saúde mental, com diferentes gêneros musicais influenciando de maneira distinta o bem-estar emocional dos indivíduos. As abordagens variam desde o uso de algoritmos clássicos de classificação até modelos mais complexos como o Gaussian Process, cada um com suas vantagens e limitações em termos de precisão e aplicabilidade. Uma tendência comum é a utilização de técnicas de machine learning para identificar padrões e prever o impacto da música na saúde mental, destacando o potencial da musicoterapia como uma intervenção acessível e eficaz.

4. Conclusão do Estado da Arte
A revisão da literatura sugere que a música, quando usada de forma estratégica, pode ser uma ferramenta poderosa para melhorar a saúde mental. Os estudos analisados oferecem uma base sólida para futuras pesquisas e aplicações no campo da musicoterapia, especialmente em contextos de saúde pública. A integração de técnicas de machine learning com análises de dados musicais proporciona insights valiosos que podem ser utilizados para desenvolver intervenções personalizadas, contribuindo para o avanço da saúde mental global.

# Descrição do _dataset_ selecionado

Nesta seção, você deverá descrever detalhadamente o _dataset_ selecionado. Lembre-se de informar o link de acesso a ele, bem como, de descrever cada um dos seus atributos (a que se refere, tipo do atributo etc.), se existem atributos faltantes etc.

# Canvas analítico

Nesta seção, você deverá estruturar o seu Canvas Analítico. O Canvas Analítico tem o papel de registrar a organização das ideias e apresentar o modelo de negócio. O Canvas Analítico deverá ser preenchido integralmente mesmo que você não tenha "tantas certezas".

> **Links Úteis**:
> - [Modelo do Canvas Analítico](https://github.com/ICEI-PUC-Minas-PMV-SI/PesquisaExperimentacao-Template/blob/main/help/Software-Analtics-Canvas-v1.0.pdf)

# Referências

<!-- Inclua todas as referências (livros, artigos, sites, etc) utilizados no desenvolvimento do trabalho utilizando o padrão ABNT.

> **Links Úteis**:
> - [Padrão ABNT PUC Minas](https://portal.pucminas.br/biblioteca/index_padrao.php?pagina=5886) -->

* Organização Mundial da Saúde (OMS): <br/>
**Fonte**: World Health Organization (WHO). "Mental health and COVID-19: early evidence of the pandemic’s impact." WHO, 2022. <br/>
**Disponível em**: [WHO Mental Health.](https://www.who.int/news/item/02-03-2022-covid-19-pandemic-triggers-25-increase-in-prevalence-of-anxiety-and-depression-worldwide) <br/>

* Ministério da Saúde do Brasil: <br/>
Estimativa sobre a porcentagem da população adulta brasileira diagnosticada com transtornos mentais comuns, como ansiedade e depressão. <br/>
**Fonte**: Ministério da Saúde. "Saúde Mental no SUS: As Redes de Atenção Psicossocial." Brasília: Ministério da Saúde, 2013. <br/>
**Disponível em**: [Ministério da Saúde - Saúde Mental.](https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/s/saude-mental) <br/>

* Eficácia da musicoterapia no tratamento de sintomas de transtornos mentais como ansiedade, depressão, insônia e TOC. <br/>
**Fonte**: Bradt, Joke, et al. "Music interventions for improving psychological and physical outcomes in cancer patients." Cochrane Database of Systematic Reviews 2016, Issue 8. Art. No.: CD006911. DOI: 10.1002/14651858.CD006911.pub3. <br/>
**Fonte**: Maratos, Anna, et al. "Music therapy for depression." Cochrane Database of Systematic Reviews 2008, Issue 1. Art. No.: CD004517. DOI: 10.1002/14651858.CD004517.pub2. <br/>

* Music Classification and Mental Health Analysis using Exploratory Data Analysis
*Fonte**: BHAVANI, V.; SRAVANI, K.; SIRIVARSHITHA, A. K.; PRIYA, K. S. Music Classification and Mental Health Analysis using Exploratory Data Analysis. In: 2023 International Conference on Innovative Data Communication Technologies and Application (ICIDCA-2023), Vaddeswaram, AP, India, 2023. p. 555-561. DOI: 10.1109/ICIDCA56705.2023.10099605.
