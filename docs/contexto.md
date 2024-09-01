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
<br/><br/>"O quão efetivo cada genero musical é na redução dos sintomas de depressão, ansiedade, insônia e TOC?"<br/><br/> 
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

Analisando o artigo "Music Classification and Mental Health Analysis using Exploratory Data Analysis" temos os seguintes resultados: <br/>

1. Contextualização do Problema <br/>
O estudo aborda a relação entre o consumo de diferentes gêneros musicais e seus impactos na saúde mental, com foco em condições como ansiedade, depressão, insônia e transtorno obsessivo-compulsivo (TOC). O problema é contextualizado a partir do entendimento de que a música pode influenciar emoções humanas e, por consequência, a saúde mental. O objetivo principal é analisar como diferentes tipos de música podem afetar essas condições, possivelmente auxiliando no tratamento ou compreensão dos estados mentais dos indivíduos.(BHAVANI et al., 2023) <br/>

2. Características do Dataset Utilizado <br/>
O dataset foi coletado através de um formulário do Google, distribuído por redes sociais, servidores de Discord e diversos sites no Reddit. Ele contém 736 registros e 33 colunas, sem restrições de idade ou localização dos respondentes. As colunas incluem variáveis como idade, gênero musical preferido, horas diárias de música ouvidas, e severidade de condições mentais (medida em uma escala de 0 a 10 para ansiedade, depressão, insônia e TOC). Antes de ser utilizado na análise, o dataset foi pré-processado para remover valores nulos, outliers, e observações irrelevantes.(BHAVANI et al., 2023)​ <br/>

3.  Abordagens/Algoritmos Utilizados: <br/>
O estudo utilizou a análise exploratória de dados (EDA) como principal abordagem para identificar padrões e correlações no dataset. Além disso, o algoritmo de classificação Gaussian Naive Bayes foi aplicado para prever como diferentes gêneros musicais afetam a saúde mental. Este algoritmo é escolhido devido à sua adequação para dados contínuos, assumindo que cada parâmetro pode prever o resultado de forma independente. Os passos incluem a divisão do dataset em componentes de treino e teste, e a aplicação do Gaussian Naive Bayes para modelar as predições.(BHAVANI et al., 2023) <br/>

4. Métricas de Avaliação Empregadas <br/>
A principal métrica utilizada para avaliar o desempenho do modelo foi a acurácia, que reflete a proporção de predições corretas feitas pelo modelo. No estudo, a acurácia alcançada pelo modelo Gaussian Naive Bayes foi de 68,2% utilizando Python, e 74% ao utilizar o software RapidMiner, que é capaz de lidar com grandes volumes de dados e oferece pacotes integrados e configuráveis para análise.(BHAVANI et al., 2023) <br/>

5. Resultados Obtidos <br/>
Os resultados indicam que o consumo de música está correlacionado com diferentes níveis de condições mentais. Por exemplo, indivíduos que escutam gêneros como rock, jazz, K-pop, hip-hop, e pop apresentam níveis mais altos de ansiedade. Já a insônia é menos prevalente entre ouvintes de metal, lofi e gospel. Além disso, a maioria dos ouvintes de rap e lofi tem níveis de TOC mais elevados, e gêneros como rock, hip-hop e lofi estão associados a níveis mais altos de depressão. A acurácia obtida sugere que o modelo é razoavelmente eficaz na previsão dos efeitos da música sobre a saúde mental, embora haja espaço para melhorias com a adição de mais dados e variáveis demográficas. (BHAVANI et al., 2023) <br/>

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
