# Introdução

<!-- Texto descritivo introdutório apresentando a visão geral do projeto a ser desenvolvido considerando o contexto em que ele se insere, os objetivos gerais, a justificativa e o público-alvo do projeto. -->

A saúde mental é um componente essencial para o bem-estar do ser humano, influenciando profundamente aspectos físicos, sociais e emocionais da vida. 

No cenário global atual, a prevalência de transtornos mentais tem se tornado uma questão alarmante, especialmente após os impactos da pandemia de COVID-19. Segundo a Organização Mundial da Saúde (OMS, 2022), no Brasil, estima-se que 9,3% da população adulta enfrente desafios relacionados à saúde mental, como ansiedade e depressão. (BRASIL, 2013, p. 25).

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

A saúde mental é um dos pilares essenciais para a qualidade de vida, influenciando diretamente o bem-estar físico, social e emocional das pessoas. Estudos recentes indicam que uma parcela significativa da população mundial sofre de algum tipo de transtorno mental. De acordo com a Organização Mundial da Saúde (OMS, 2022), cerca de 1 em cada 8 pessoas no mundo vive com um transtorno mental, número que se agravou significativamente após a pandemia de COVID-19, que gerou um aumento de 25% na prevalência de ansiedade e depressão em nível global. No Brasil, estima-se que aproximadamente 9,3% da população adulta tenha sido diagnosticada com algum transtorno mental comum, como ansiedade ou depressão (BRASIL, 2013, p. 25).

Diante desse cenário alarmante, torna-se imperativo buscar alternativas complementares aos tratamentos tradicionais, muitas vezes centrados no uso de medicamentos. Uma dessas alternativas é a música, cuja influência positiva na saúde mental tem sido amplamente discutida. A musicoterapia, por exemplo, tem demonstrado eficácia na redução dos sintomas de ansiedade, depressão, insônia e Transtorno Obsessivo-Compulsivo (TOC) (BRADT et al., 2016; MARATOS et al., 2008). Além disso, estudos indicam que a exposição ao ruído branco pode ser benéfica para melhorar o sono em crianças, ajudando a acalmar e reduzir a ansiedade (SPENCER et al., 1990). A música também tem sido associada a melhorias no bem-estar físico e mental de pacientes hospitalizados, contribuindo para a recuperação mais rápida e a diminuição do estresse (ULRICH et al., 1991).

O presente projeto de dados visa explorar a relação entre o consumo de música através de aplicativos e a melhora nos sintomas de doenças mentais, com base em uma base de dados que combina informações de usuários e sua autoavaliação sobre os efeitos da música. A escolha desse tema se fundamenta na crescente necessidade de encontrar soluções acessíveis e não invasivas que possam complementar os tratamentos convencionais, oferecendo alívio e melhor qualidade de vida para aqueles que sofrem de transtornos mentais.

O impacto social desse estudo é potencialmente significativo, considerando que o acesso à música é amplamente democratizado e pode ser facilmente integrado ao cotidiano das pessoas. Analisando a eficácia da música como ferramenta terapêutica, pode-se promover a sua utilização como um meio acessível e eficaz para a mitigação dos sintomas de transtornos mentais, contribuindo assim para a saúde pública e para a redução do estigma associado a esses transtornos.

A relevância deste projeto, portanto, reside na possibilidade de oferecer uma abordagem inovadora e de baixo custo para o tratamento de condições de saúde mental, ao mesmo tempo em que se fortalece a compreensão do papel da música no bem-estar humano.

## Público-Alvo

<!-- Descreva quem serão as pessoas que poderão se beneficiar com a sua investigação indicando os diferentes perfis. O objetivo aqui não é definir quem serão os clientes ou quais serão os papéis dos usuários na aplicação. A ideia é, dentro do possível, conhecer um pouco mais sobre o perfil dos usuários: conhecimentos prévios, relação com a tecnologia, relações hierárquicas, etc.

Adicione informações sobre o público-alvo por meio de uma descrição textual, diagramas de personas e mapa de stakeholders. -->

O estudo beneficiará pessoas de várias idades, desde adolescentes até idosos, que têm diferentes graus de familiaridade com tecnologia e diferentes motivações para o uso de música em suas vidas. O projeto pode fornecer insights valiosos para profissionais de saúde mental sobre como personalizar intervenções musicais de acordo com o perfil etário e tecnológico dos pacientes. Além disso, pode orientar desenvolvedores de aplicativos de música a criar funcionalidades que melhor atendam às necessidades de seus usuários, promovendo bem-estar e saúde mental.

* Embora a base de dados utilizada possua avaliação de usuários de todas as idades, mais de 70% da base se concentra na faixa etária entre 14 e 27 anos.

<!-- > **Links Úteis**:
> - [Público-alvo](https://blog.hotmart.com/pt-br/publico-alvo/)
> - [Como definir o público alvo](https://exame.com/pme/5-dicas-essenciais-para-definir-o-publico-alvo-do-seu-negocio/)
> - [Público-alvo: o que é, tipos, como definir seu público e exemplos](https://klickpages.com.br/blog/publico-alvo-o-que-e/)
> - [Qual a diferença entre público-alvo e persona?](https://rockcontent.com/blog/diferenca-publico-alvo-e-persona/) -->

## Estado da arte

<!--Nesta seção, deverão ser descritas outras abordagens identificadas na literatura que foram utilizadas para resolver problemas similares ao problema em questão. Para isso, faça uma pesquisa detalhada e identifique, no mínimo, 5 trabalhos que tenham utilizado dados em contexto similares e então: (a) detalhe e contextualize o problema, (b) descreva as principais características do _dataset_ utilizado, (c) detalhe quais abordagens/algoritmos foram utilizados (e seus parâmetros), (d) identifique as métricas de avaliação empregadas, e (e) fale sobre os resultados obtidos. 

> **Links Úteis**:
> - [Google Scholar](https://scholar.google.com/)
> - [IEEE Xplore](https://ieeexplore.ieee.org/Xplore/home.jsp)
> - [Science Direct](https://www.sciencedirect.com/)
> - [ACM Digital Library](https://dl.acm.org/) -->

O primeiro trabalho utilizado como referência, descreve um sistema de recomendação de músicas baseado em emoções, desenvolvido em resposta ao aumento de distúrbios emocionais durante a pandemia de COVID-19. O sistema utiliza algoritmos como Random Forest e XGBoost para classificar emoções de músicas e recomendar faixas que se alinhem com o estado emocional do usuário. Os dados incluem características de áudio do Spotify e letras de músicas, com uma precisão de até 85% nas recomendações.

**1. Contextualização do Problema:**
O problema abordado é a necessidade de um sistema de recomendação de músicas baseado em emoções, especialmente devido ao impacto da pandemia de COVID-19, que levou a um aumento nos distúrbios de humor, como depressão e ansiedade. A música é vista como um potencial companheiro empático para ajudar as pessoas durante esses tempos difíceis, e o sistema proposto utiliza a emoção do usuário como entrada para recomendar músicas que se alinhem com seu estado emocional.

**2. Características do Dataset Utilizado:**
O sistema utiliza um conjunto de dados de emoções, extraído de um projeto open-source no GitHub, combinado com o Spotify Dataset, que contém recursos de áudio de músicas lançadas entre 1922 e 2021. Além disso, utiliza o Million Song Dataset, um conjunto de dados de características e metadados de músicas. As características consideradas incluem 'valance', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', e 'speechiness'.

**3. Abordagens/Algoritmos Utilizados:**
Random Forest: Utilizado para o treinamento inicial, com o número de árvores (n_estimators) fixado em 20 e o critério de divisão como "Gini".<br/>

XGBoost: Utilizado para classificar as emoções das músicas. Parâmetros incluíram learning_rates (0.1, 0.2, 0.5), max_depth (5, 10, 15), n_estimators (150, 250, 300), e min_child_weight (3, 5, 10).<br/>

K-Means Clustering: Foi também utilizado para comparar o desempenho de modelos supervisionados e não supervisionados.<br/>

TF-IDF (Term Frequency-Inverse Document Frequency): Usado para processar as letras das músicas e gerar uma matriz de similaridade para melhorar as recomendações.

**4. Métricas de Avaliação Empregadas**

As métricas utilizadas para avaliar o desempenho do sistema de recomendação de músicas baseado em emoções incluem:

Acurácia: A acurácia foi a principal métrica de avaliação mencionada, que variou entre 77% e 85% para os modelos de classificação utilizados (Random Forest e XGBoost). A acurácia mede a proporção de predições corretas feitas pelo modelo em relação ao total de predições.

Matriz de Confusão: Embora não seja explicitamente detalhado no trecho fornecido, a matriz de confusão é mencionada como uma ferramenta utilizada para avaliar o desempenho dos modelos de classificação. A matriz de confusão permite uma análise mais detalhada das predições corretas e incorretas, mostrando o número de verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.

**5. Resultados Obtidos**
O sistema foi capaz de recomendar músicas com uma precisão que variava entre 77% e 85%. As recomendações foram baseadas tanto nas emoções identificadas nas músicas quanto na similaridade lírica. O sistema conseguiu identificar emoções com um bom nível de precisão, e os resultados observacionais indicaram que as recomendações baseadas em emoção proporcionaram uma experiência mais satisfatória para os usuários.

Esses resultados sugerem que o sistema proposto é eficaz para recomendar músicas baseadas nas emoções do usuário e pode ser integrado em qualquer mecanismo de recomendação de música existente.

**Padrões na preferência musical dos brasileiros sob a ótica do Spotify**

O segundo trabalho, "Padrões na preferência musical dos brasileiros sob a ótica do Spotify" busca analisar e compreender os comportamentos e tendências musicais no Brasil a partir dos dados disponíveis na plataforma Spotify.

**1. Contextualização do Problema:**
O problema central envolve a identificação de padrões e preferências musicais entre os brasileiros, considerando fatores como gênero musical, frequência de reprodução, e temporalidade, a fim de entender como esses padrões variam entre diferentes grupos demográficos.

**2. Características do Dataset Utilizado:**
O conjunto de dados utilizado é oriundo do Spotify e inclui informações detalhadas sobre faixas de música, gêneros, artistas, datas de reprodução, e dados demográficos dos usuários. Este dataset é volumoso e rico, permitindo análises profundas sobre as preferências musicais ao longo do tempo e entre diferentes grupos de usuários.

**3. Abordagens/Algoritmos Utilizados:**
Foram utilizados métodos de análise exploratória de dados, segmentação de usuários via clustering (ex: K-means), e algoritmos de machine learning como Regressão Logística e Random Forest para prever padrões de preferência e recomendação musical. Parâmetros como o número de clusters no K-means foram otimizados para maximizar a homogeneidade dentro dos clusters.

**4. Métricas de Avaliação Empregadas**
As métricas de avaliação incluem a acurácia, precisão, recall e F1-score para avaliar o desempenho dos modelos preditivos. Além disso, a análise de variância (ANOVA) foi usada para testar a significância das diferenças entre os grupos identificados.

**5. Resultados Obtidos**
Os resultados indicam que certos gêneros musicais têm uma forte associação com determinados grupos demográficos, e que a temporalidade, como o dia da semana e a hora do dia, afeta significativamente os padrões de escuta. Além disso, os modelos preditivos demonstraram bom desempenho em prever as preferências musicais, com as abordagens baseadas em Random Forest obtendo os melhores resultados.

O terceiro artigo utilizado como referência, intitulado "Efeitos da musicoterapia sobre os sintomas de ansiedade e depressão em adultos com diagnóstico de transtornos mentais: revisão sistemática", explora o impacto da musicoterapia como uma intervenção terapêutica para aliviar sintomas de ansiedade e depressão em adultos. Através de uma revisão sistemática de ensaios clínicos randomizados, o estudo busca identificar e sintetizar as evidências sobre a eficácia dessa abordagem, fornecendo uma análise detalhada de como a musicoterapia pode melhorar o bem-estar mental dos pacientes em diferentes contextos clínicos.

**1. Contextualização do Problema:**
O artigo "Efeitos da musicoterapia sobre os sintomas de ansiedade e depressão em adultos com diagnóstico de transtornos mentais: revisão sistemática" aborda a crescente preocupação com os transtornos mentais, como ansiedade e depressão, que afetam cerca de 700 milhões de pessoas globalmente. Essas condições são particularmente prevalentes em adultos e têm um impacto devastador na qualidade de vida, especialmente em pacientes com transtornos mentais. A musicoterapia é discutida como uma intervenção complementar promissora que pode aliviar esses sintomas, melhorando o bem-estar físico e mental dos pacientes. A revisão sistemática realizada no estudo busca sintetizar as evidências disponíveis sobre os efeitos da musicoterapia especificamente em adultos diagnosticados com transtornos mentais, oferecendo uma visão crítica sobre a eficácia dessa abordagem terapêutica​​.

**2. Características do Dataset Utilizado:**
O dataset utilizado na revisão foi composto por 1.649 estudos identificados em bases de dados como MEDLINE, Embase, CENTRAL Cochrane, CINAHL, PsycINFO e LILACS. Destes, após um rigoroso processo de triagem e avaliação metodológica, apenas oito estudos foram selecionados para análise detalhada. Esses estudos foram realizados em diferentes países, incluindo Brasil, Canadá, Finlândia, França, Irã, Noruega, Coreia do Sul e Estados Unidos, com amostras que variaram de 26 a 113 participantes adultos diagnosticados com transtornos mentais. As intervenções de musicoterapia analisadas incluíam tanto abordagens ativas quanto passivas, com diferentes durações e frequências de sessões.

**3. Abordagens/Algoritmos Utilizados:**
A revisão sistemática utilizou uma metodologia rigorosa para selecionar e analisar os ensaios clínicos randomizados (ECR). A qualidade metodológica dos estudos incluídos foi avaliada por meio da Escala de Jadad, que classifica os estudos em uma pontuação de 0 a 5, sendo os estudos com pontuação maior ou igual a 3 considerados de alta qualidade. Além disso, foi utilizada a ferramenta de Risco de Viés da Cochrane (RoB 1) para avaliar a validade interna e o risco de viés dos estudos. Os principais domínios de avaliação incluíam a alocação da sequência de randomização, cegamento dos participantes e equipe, desfechos incompletos, e relato seletivo de desfechos.

**4. Métricas de Avaliação Empregadas**
As métricas de avaliação incluíram principalmente a Escala Hospitalar de Ansiedade e Depressão (HADS), utilizada em cinco dos oito estudos para medir os níveis de ansiedade e depressão. Outros instrumentos incluíram o State-Trait Anxiety Inventory (STAI), o Beck Depression Inventory (BDI), a Visual Analog Scale (VAS) para dor, e o Profile of Mood States (POMS) para avaliar o humor. A qualidade dos estudos foi também mensurada pela Escala de Jadad e pelo risco de viés segundo a Cochrane, com quatro estudos classificados como de baixo risco de viés e os demais com alto risco.

**5. Resultados Obtidos**
Os resultados indicam que a musicoterapia tem um efeito positivo significativo na redução dos sintomas de ansiedade e depressão em adultos com transtornos mentais. Os pacientes que participaram das intervenções de musicoterapia apresentaram relaxamento físico e mental, bem como uma redução significativa dos sintomas ansiosos e depressivos, promovendo o bem-estar geral. Quatro dos estudos analisados foram classificados como de alta qualidade metodológica e baixo risco de viés, o que fortalece a validade dos achados. Contudo, a revisão também apontou para a necessidade de mais estudos com amostras maiores e metodologias mais robustas para elucidar completamente os mecanismos subjacentes e potencializar os efeitos benéficos da musicoterapia.

**Robot Assisted Music Therapy: A Case Study with Children Diagnosed with Autism**

O quarto trabalho "Robot Assisted Music Therapy: A Case Study with Children Diagnosed with Autism" busca analisar e compreender os efeitos da terapia musical assistida por robôs em crianças diagnosticadas com autismo. Utilizando o robô NAO durante as sessões de musicoterapia, o estudo incentivou as crianças a imitarem movimentos de dança sincronizados com músicas cuidadosamente selecionadas pelos terapeutas, promovendo uma abordagem inovadora para o desenvolvimento de habilidades sociais e de imitação.

**1. Contextualização do Problema:**
O problema central busca encontrar maneiras mais eficazes e amigáveis de ajudar crianças com autismo a melhorar sua interação social através de imitação e interação social. Este estudo investiga como a terapia musical, quando combinada com o robô NAO, pode tornar as sessões de musicoterapia mais dinâmicas e engajantes. A ideia é fazer com que o robô incentive as crianças a imitarem movimentos de dança enquanto os terapeutas tocam as músicas que eles escolheram. Isso torna o ambiente de aprendizado mais divertido e motivador. O estudo também procura entender como essa abordagem pode ajudar os terapeutas a trabalhar melhor, permitindo-lhes ver melhor as respostas das crianças e ajustar as intervenções de acordo com as necessidades de cada uma, resultando em um cuidado mais personalizado. 

**2. Características do Dataset Utilizado:**
O estudo usa um robô NAO para coletar dados de sessões de musicoterapia com quatro crianças com autismo. A análise de vídeo capturou informações comportamentais para o dataset. Essas informações incluem a frequência com que as crianças imitavam os movimentos de dança do robô e a frequência com que os terapeutas intervieram. As sessões de terapia ocorreram por pelo menos seis semanas, e cada uma delas foi registrada para análise. O tipo e a frequência dos movimentos imitados, a quantidade de prompts fornecidos pelos terapeutas, o tempo de resposta das crianças e o contexto das sessões (local e música utilizada) são algumas das características consideradas no dataset. Esses fatores foram utilizados para avaliar o impacto do robô na melhoria das habilidades de imitação e interação social das crianças ao longo do tempo. 

**3. Abordagens/Algoritmos Utilizados:**
Foram utilizados métodos de análise de comportamento, especificamente a observação da frequência de imitação de movimentos de dança por crianças diagnosticadas com autismo, quando instruídas por um robô NAO durante sessões de musicoterapia. A análise dos dados comportamentais foi feita por meio de vídeos gravados das sessões, avaliando o número de vezes que as crianças imitaram os movimentos do robô e a frequência de intervenções do terapeuta. Não foram mencionados algoritmos de aprendizado de máquina ou técnicas avançadas de clustering. A abordagem focou principalmente na interação humano-robô e na análise qualitativa das respostas das crianças. 

**4. Métricas de Avaliação Empregadas**

Frequência de Imitação: A principal métrica de avaliação utilizada foi a frequência com que as crianças imitavam os movimentos de dança dos robôs durante as sessões de musicoterapia. Ao longo das seis semanas do estudo, essa métrica foi medida em cada sessão. Os resultados mostraram que a imitação dos movimentos aumentou, passando de uma média de 12,5 movimentos na primeira semana para 21,75 na sexta semana. 

Frequência de Intervenções do Terapeuta: Outra métrica importante foi a frequência de intervenções ou prompts dados pelos terapeutas para incentivar a imitação dos movimentos do robô. Essa métrica foi utilizada para avaliar o grau de independência das crianças ao seguir as instruções do robô, com os resultados indicando uma redução nas intervenções de uma média de 43,5 na primeira semana para 28,75 na sexta semana. 

Análise de Concordância entre Avaliadores: Para garantir a confiabilidade na coleta de dados, foi empregada uma métrica de concordância entre dois avaliadores independentes que analisaram os vídeos das sessões. A concordância alcançada foi de 82%, assegurando que a avaliação das imitações e intervenções fosse consistente. 

Evolução Temporal: A análise da evolução temporal das métricas de frequência de imitação e intervenções foi utilizada para avaliar as tendências de melhora ao longo do tempo, oferecendo uma visão longitudinal do progresso das crianças. 

**5. Resultados Obtidos**
Os resultados do estudo mostraram que a terapia assistida por robôs pode ser uma abordagem eficaz para melhorar a capacidade de imitação em crianças diagnosticadas com autismo. Observou-se que, ao longo de seis semanas de sessões de musicoterapia, houve um aumento significativo na frequência com que as crianças imitavam os movimentos de dança demonstrados pelo robô. No início do estudo, as crianças imitavam uma média de 12,5 movimentos por sessão, e esse número subiu para uma média de 21,75 movimentos na sexta semana. 

Além disso, o estudo revelou uma redução no número de intervenções dos terapeutas à medida que as sessões progrediam. No início, os terapeutas precisavam intervir em média 43,5 vezes por sessão para incentivar as crianças a imitar os movimentos do robô. Esse número diminuiu para uma média de 28,75 intervenções por sessão na última semana do estudo. Isso sugere que o uso do robô não apenas ajudou a aumentar a imitação, mas também permitiu que os terapeutas se concentrassem mais na observação das crianças e menos em guiar diretamente suas ações. 

Os resultados também indicaram que a resposta ao uso do robô variou entre as crianças, mostrando que algumas crianças responderam melhor à terapia do que outras. Isso destaca a importância de adaptar a terapia às necessidades individuais de cada criança. De modo geral, os resultados sugerem que o robô NAO pode ser uma ferramenta valiosa para apoiar o desenvolvimento de habilidades sociais e de imitação em crianças com autismo durante a musicoterapia. 

**Make Your Favorite Music Curative: Music Style Transfer for Anxiety Reduction**

O quinto estudo tem como premissa transformar a música que as pessoas ouvem no dia-a-dia em música terapêutica, o que faz com que seja uma forma de musicoterapia, porém de forma bem mais simples, na qual uma pessoa que já ouve muitas horas de música no dia teria bastante facilidade de incorporar na sua rotina.

**1. Contextualização do Problema:**
Nos últimos anos, cada vez mais o tema saúde mental vem recebendo atenção quando se fala em saúde, onde em 2017, 300 milhões de pessoas foram afetadas com sintomas de ansiedade (OMS, 2017). Para auxiliar nessa questão, a música é uma das utilizadas ajudar as pessoas nessa jornada, e com base nessa premissa, o estudo foca em utilizar as músicas favoritas das pessoas que participaram, e transforma-las em músicas focadas no tratamento de doenças mentais.	

**2. Características do Dataset Utilizado:**
O dataset utilizado consiste em dados de músicas que o usuário gosta e dados de músicas terapêuticas as quais foram escolhidas por terapeutas.

**3. Abordagens/Algoritmos Utilizados:**
Primeiro, foi feito um algoritmo para verificar quão parecido/diferente é cada gênero musical através de um algoritmo que consegue pegar traços marcantes de cada gênero musical. Depois foi feito um teste para ver quão rápido foi executado todo o processo de conversão de música em música terapêutica, e o resultado obtido nesse teste é que o modelo proposto foi feito em um período razoável, apesar de alguns resultados terem ficado com ruídos ao invés de sair como uma música. A última avaliação do modelo proposto, foi o teste com pessoas, na qual durante quatro dias elas ouviriam ou suas músicas preferidas, ou música terapêutica, ou a música gerada pelo modelo, ou não ouviriam nenhuma para controle dos testes, e após a sessão de música, iriam responder um questionário.

**4. Métricas de Avaliação Empregadas**
Para os testes de performance do modelo, foram utilizados algoritmos de transformação de imagem para conseguirem comparar o quão eficiente estava o modelo, e no teste com pessoas, utilizaram o State Train Anxiety Inventory(STAI, Form Y Version) para medir os resultados do dia de teste após ouvirem as músicas

**5. Resultados Obtidos**
Após a apuração da última etapa de testes, foi relatado que as pessoas se sentem constantemente melhores dos sintomas da ansiedade após ouvirem músicas transformadas pelo modelo, e também pelas músicas terapêutica, e apesar dos bons resultados, esse é apenas o primeiro estudo direcionado a transformação de música para música de terapia e há muito espaço para algoritmos e modelos de deep learning para melhores resultados futuramente.



# Descrição do _dataset_ selecionado 

Nesta seção, você deverá descrever detalhadamente o _dataset_ selecionado. Lembre-se de informar o link de acesso a ele, bem como, de descrever cada um dos seus atributos (a que se refere, tipo do atributo etc.), se existem atributos faltantes etc.

O dataset escolhido possui 736 entradas armazenadas, todas em uma única string, separadas pelo delimitador vírgula (","), em um arquivo do tipo .csv. As colunas foram devidamente separadas para permitir uma análise adequada do dataset escolhido.

De modo resumido, o dataset possui um total de 736 registros, 33 atributos.

As colunas Age, Primary streaming service, While working, Instrumentalist, Composer, Foreign languages, BPM, Music effects, possuem alguns dados faltantes. 

Link para acesso do Dataset escolhido:

https://kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results?select=mxmh_survey_results.csv


**1.Timestamp:** 
Descrição: Data e hora em que a entrada foi registrada.<br/>
Tipo: object (Texto, mas deve ser convertido para datetime para análises de tempo).

**2. Age:**
Descrição: Idade do participante.<br/>
Tipo: object (Texto, mas deve ser convertido para int).

**3. Primary streaming service:**
Descrição: Serviço de streaming de música mais utilizado pelo participante (e.g., Spotify, YouTube Music).<br/>
Tipo: object (Texto).

**4. Hours per day:**
Descrição: Quantidade de horas diárias que o participante ouve música.<br/>
Tipo: object (Texto, mas deve ser convertido para float).

**5. While working:**
Descrição: Se o participante ouve música enquanto trabalha (Sim/Não).<br/>
Tipo: object (Texto).

**6. Instrumentalist:**
Descrição: Se o participante toca algum instrumento (Sim/Não).<br/>
Tipo: object (Texto).

**7. Composer:**
Descrição: Se o participante compõe música (Sim/Não).<br/>
Tipo: object (Texto).

**8. Fav genre:**
Descrição: Gênero musical favorito do participante.<br/>
Tipo: object (Texto).

**9. Exploratory:**
Descrição: Se o participante gosta de explorar novos gêneros musicais (Sim/Não).<br/>
Tipo: object (Texto).

**10. Foreign languages:**
Descrição: Se o participante ouve músicas em línguas estrangeiras (Sim/Não).<br/>
Tipo: object (Texto).

**11. BPM:**
Descrição: Batidas por minuto (BPM) preferidas nas músicas que o participante ouve.<br/>
Tipo: object (Texto, mas deve ser convertido para int).

**12. Frequency [Classical]:**
Descrição: Frequência com que o participante ouve música clássica.<br/>
Tipo: object (Texto).

**13. Frequency [Country]:**
Descrição: Frequência com que o participante ouve música country.<br/>
Tipo: object (Texto).

**14. Frequency [EDM]:**
Descrição: Frequência com que o participante ouve música eletrônica (EDM).<br/>
Tipo: object (Texto).

**15. Frequency [Folk]:**
Descrição: Frequência com que o participante ouve música folk.<br/>
Tipo: object (Texto).

**16. Frequency [Gospel]:**
Descrição: Frequência com que o participante ouve música gospel.<br/>
Tipo: object (Texto).

**17. Frequency [Hip hop]:**
Descrição: Frequência com que o participante ouve hip hop.<br/>
Tipo: object (Texto).

**18. Frequency [Jazz]:**
Descrição: Frequência com que o participante ouve jazz.<br/>
Tipo: object (Texto).

**19. Frequency [K pop]:**
Descrição: Frequência com que o participante ouve K-pop.<br/>
Tipo: object (Texto).

**20. Frequency [Latin]:**
Descrição: Frequência com que o participante ouve música latina.<br/>
Tipo: object (Texto).

**21. Frequency [Lofi]:**
Descrição: Frequência com que o participante ouve música lofi.<br/>
Tipo: object (Texto).

**22. Frequency [Metal]:**
Descrição: Frequência com que o participante ouve música metal.<br/>
Tipo: object (Texto).

**23. Frequency [Pop]:**
Descrição: Frequência com que o participante ouve música pop.<br/>
Tipo: object (Texto).

**24. Frequency [R&B]:**
Descrição: Frequência com que o participante ouve R&B.<br/>
Tipo: object (Texto).

**25. Frequency [Rap]:**
Descrição: Frequência com que o participante ouve rap.<br/>
Tipo: object (Texto).

**26. Frequency [Rock]:**
Descrição: Frequência com que o participante ouve rock.<br/>
Tipo: object (Texto).

**27. Frequency [Video game music]:**
Descrição: Frequência com que o participante ouve músicas de jogos eletrônicos.<br/>
Tipo: object (Texto).

**28. Anxiety:**
Descrição: Nível de ansiedade do participante (escala de 0 a 10).<br/>
Tipo: object (Texto, mas deve ser convertido para int).

**29. Depression:**
Descrição: Nível de depressão do participante (escala de 0 a 10).<br/>
Tipo: object (Texto, mas deve ser convertido para int).

**30. Insomnia:**
Descrição: Nível de insônia do participante (escala de 0 a 10).<br/>
Tipo: object (Texto, mas deve ser convertido para int).

**31. OCD:**
Descrição: Nível de Transtorno Obsessivo-Compulsivo (OCD) do participante (escala de 0 a 10).<br/>
Tipo: object (Texto, mas deve ser convertido para int).

**32. Music effects:**
Descrição: Efeitos percebidos da música na saúde mental do participante (e.g., Improve, Worsen, No effect).<br/>
Tipo: object (Texto).

**33. Permissions:**
Descrição: Consentimento do participante para o uso dos dados.<br/>
Tipo: object (Texto).

# Canvas analítico

<!--Nesta seção, você deverá estruturar o seu Canvas Analítico. O Canvas Analítico tem o papel de registrar a organização das ideias e apresentar o modelo de negócio. O Canvas Analítico deverá ser preenchido integralmente mesmo que você não tenha "tantas certezas".

> **Links Úteis**:
> - [Modelo do Canvas Analítico](https://github.com/ICEI-PUC-Minas-PMV-SI/PesquisaExperimentacao-Template/blob/main/help/Software-Analtics-Canvas-v1.0.pdf) -->

> - [Canvas Analítico Música e Saúde Mental]([https://github.com/ICEI-PUC-Minas-PMV-SI/PesquisaExperimentacao-Template/blob/main/help/Canvas analitico Música e Saúde Mental (2).pdf](https://github.com/ICEI-PUC-Minas-PMV-SI/pmv-si-2024-2-pe7-t1-musicaesaudemental/blob/main/help/Canvas%20analitico%20M%C3%BAsica%20e%20Sa%C3%BAde%20Mental%20(2).pdf))

# Referências

<!-- Inclua todas as referências (livros, artigos, sites, etc) utilizados no desenvolvimento do trabalho utilizando o padrão ABNT.

> **Links Úteis**:
> - [Padrão ABNT PUC Minas](https://portal.pucminas.br/biblioteca/index_padrao.php?pagina=5886) -->

* Organização Mundial da Saúde (OMS). <br/>
**Fonte**: World Health Organization (WHO). "Depression and Other Common Mental Disorders: global health estimates" WHO, 2017. <br/>
**Disponível em**: [WHO Mental Health.]https://www.who.int/publications/i/item/depression-global-health-estimates<br/>
**Fonte**: World Health Organization (WHO). "Mental health and COVID-19: early evidence of the pandemic’s impact." WHO, 2022. <br/>
**Disponível em**: [WHO Mental Health.](https://www.who.int/news/item/02-03-2022-covid-19-pandemic-triggers-25-increase-in-prevalence-of-anxiety-and-depression-worldwide) <br/>

* Ministério da Saúde do Brasil. <br/>
Estimativa sobre a porcentagem da população adulta brasileira diagnosticada com transtornos mentais comuns, como ansiedade e depressão. <br/>
**Fonte**: Ministério da Saúde. "Saúde Mental no SUS: As Redes de Atenção Psicossocial." Brasília: Ministério da Saúde, 2013. <br/>
**Disponível em**: [Ministério da Saúde - Saúde Mental.](https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/s/saude-mental) <br/>

* Eficácia da musicoterapia no tratamento de sintomas de transtornos mentais como ansiedade, depressão, insônia e TOC. <br/>
**Fonte**: Bradt, Joke, et al. "Music interventions for improving psychological and physical outcomes in cancer patients." Cochrane Database of Systematic Reviews 2016, Issue 8. Art. No.: CD006911. DOI: 10.1002/14651858.CD006911.pub3. <br/>
**Fonte**: Maratos, Anna, et al. "Music therapy for depression." Cochrane Database of Systematic Reviews 2008, Issue 1. Art. No.: CD004517. DOI: 10.1002/14651858.CD004517.pub2. <br/>

* Music Classification and Mental Health Analysis using Exploratory Data Analysis. <br/>
**Fonte**: BHAVANI, V.; SRAVANI, K.; SIRIVARSHITHA, A. K.; PRIYA, K. S. Music Classification and Mental Health Analysis using Exploratory Data Analysis. In: 2023 International Conference on Innovative Data Communication Technologies and Application (ICIDCA-2023), Vaddeswaram, AP, India, 2023. p. 555-561. DOI: 10.1109/ICIDCA56705.2023.10099605.

* Music Recommendation System Based on Emotion <br/>
**Fonte**: [Ieeexplore. org](https://ieeexplore.ieee.org/Xplore/home.jsp) <br/>
**Disponível em**:(https://ieeexplore.ieee.org/document/9579689)

* White noise and sleep induction. Archives of Disease in Childhood. <br/>
**Fonte**: SPENCER, Judith A. D.; MORAN, D. J.; LEE, A.; TALBERT, D. G. Archives of Disease in Childhood, v. 65, n. 1, p. 135-137, 1990. DOI: 10.1136/adc.65.1.135.

* Effects of environmental simulations and music on blood donors. <br/>
**Fonte** ULRICH, Roger S.; DIMBERG, Lennart A.; DRIVER, Brenda L. physiologic and psychological states. Journal of the American Medical Association, v. 266, n. 5, p. 641-643, 1991. DOI: 10.1001/jama.1991.03470050121036.

* IBIAPINA, Aline Raquel de Sousa; LOPES-JUNIOR, Luís Carlos; VELOSO, Lorena Uchôa Portela; COSTA, Ana Paula Cardoso; SILVA JÚNIOR, Fernando José Guedes da; SALES, Jaqueline Carvalho e Silva; MONTEIRO, Claudete Ferreira de Souza. Efeitos da musicoterapia sobre os sintomas de ansiedade e depressão em adultos com diagnóstico de transtornos mentais: revisão sistemática. Acta Paulista de Enfermagem, v. 35, p. eAPE002212, 2022. DOI: http://dx.doi.org/10.37689/acta-ape/2022AR02212.
