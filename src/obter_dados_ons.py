# ---------------------------------
# ■■■ BIBLIOTECAS ■■■
# ---------------------------------

import re
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests


# ---------------------------------
# ■■■ CONFIGURAÇÕES DO RECORTE ■■■
# ---------------------------------
# O recorte tem período e conjuntos fixos. A saída da coleta local
# fica em dados_ons/, separada do CSV de referência em data/raw/.

# ====== Fonte e período ======
DATASET = 'geracao-usina-2'
API_URL = 'https://dados.ons.org.br/api/3/action/package_show'
START_DATE = '2024-01-01'
END_DATE_EXCLUSIVE = '2026-01-01'

# ====== Seleção dos conjuntos ======
STATE = 'PE'
PLANT_TYPE = 'FOTOVOLTAICA'
SELECTED_IDS = ('CJU_PEBELM', 'CJU_PEBRI', 'CJU_PECIR')
REQUIRED_COLUMNS = {
    'din_instante', 'id_estado', 'nom_tipousina', 'nom_usina',
    'id_ons', 'val_geracao',
}

# ====== Pastas locais ======
# ------ Na pasta src/, usa a raiz do projeto como referência ------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'src' else SCRIPT_DIR
OUTPUT_DIR = PROJECT_DIR / 'dados_ons'
MONTHLY_DIR = OUTPUT_DIR / 'arquivos_mensais'


# ---------------------------------
# ■■■ CATÁLOGO E DOWNLOAD ■■■
# ---------------------------------
# O catálogo fornece o endereço de cada recurso mensal. Os Parquets
# são guardados localmente para evitar baixar o mesmo mês a cada execução.


def getMonthlyResources(
    session: requests.Session,
    months: pd.PeriodIndex) -> dict:
    """Localiza um recurso Parquet por mês no catálogo do ONS.
    Finds one Parquet resource per month in the ONS catalog.

    Parâmetros / Parameters:
        session: Sessão HTTP. / HTTP session.
        months: Meses que devem constar no catálogo. / Required months.

    Retorno / Returns:
        Recursos por mês, com nome, URL e ID. /
        Resources by month, with name, URL, and ID.
    """
    # ====== Consulta ao catálogo ======
    response = session.get(API_URL, params={'id': DATASET}, timeout=60)
    response.raise_for_status()
    catalog = response.json()

    if not catalog.get('success'):
        raise RuntimeError('A consulta ao catálogo do ONS não teve sucesso.')

    # ====== Localização dos Parquets mensais ======
    month_pattern = re.compile(
        r'geracao[_-]usina[_-]2[-_](\d{4})[-_](\d{2})',
        flags=re.IGNORECASE,
    )
    resources_by_month = {}

    for resource in catalog['result']['resources']:
        file_format = str(resource.get('format', '')).strip().upper()
        name = resource.get('name') or ''
        url = resource.get('url') or ''

        if file_format != 'PARQUET' or not url:
            continue

        match = month_pattern.search(f'{name} {url}')
        if match is None:
            continue

        year, month_number = match.groups()
        month = pd.Period(f'{year}-{month_number}', freq='M')

        if month in months:
            resources_by_month.setdefault(month, []).append({
                'name': name,
                'url': url,
                'resource_id': resource.get('id'),
            })

    # ------ Cada mês precisa ter uma fonte sem ambiguidade ------
    problems = {
        str(month): len(resources_by_month.get(month, []))
        for month in months
        if len(resources_by_month.get(month, [])) != 1
    }

    if problems:
        raise RuntimeError(
            'Esperava exatamente um Parquet por mês no catálogo do ONS. '
            f'Quantidade encontrada nos meses com problema: {problems}'
        )

    return {month: resources_by_month[month][0] for month in months}


def downloadParquet(
    session: requests.Session,
    url: str,
    month: pd.Period,
    attempts: int = 3) -> pd.DataFrame:
    """Baixa, salva e lê um Parquet mensal do ONS.
    Downloads, saves, and reads a monthly ONS Parquet file.

    Parâmetros / Parameters:
        session: Sessão HTTP. / HTTP session.
        url: Endereço direto do recurso. / Direct resource URL.
        month: Mês do recurso. / Resource month.
        attempts: Máximo de tentativas de download. / Maximum retries.

    Retorno / Returns:
        Conteúdo do Parquet. / Parquet contents as a DataFrame.
    """
    # ====== Reaproveitamento do arquivo local ======
    original_name = Path(unquote(urlparse(url).path)).name
    if not original_name:
        raise ValueError(f'Não foi possível obter o nome do arquivo: {url}')

    file_path = MONTHLY_DIR / f'{month}_{original_name}'
    if file_path.is_file():
        try:
            data = pd.read_parquet(file_path)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f'O Parquet local não pôde ser lido: {file_path}'
            ) from error

        print(f'  Reutilizando: {file_path.name}')
        return data

    # ====== Download e validação ======
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=180)
            response.raise_for_status()
            content = response.content
            data = pd.read_parquet(BytesIO(content))
            file_path.write_bytes(content)
            return data
        except (requests.RequestException, OSError, ValueError) as error:
            if attempt == attempts:
                raise RuntimeError(
                    f'Falha ao obter ou ler o Parquet de {month}: {url}'
                ) from error

            print(f'  Tentativa {attempt} falhou. Tentando novamente...')
            time.sleep(3)

    raise ValueError('O número de tentativas deve ser maior que zero.')


# ---------------------------------
# ■■■ FILTRO E CONSOLIDAÇÃO ■■■
# ---------------------------------
# Primeiro, as contagens por arquivo registram o filtro fotovoltaico
# de Pernambuco. Depois, a base final é recortada pelos três IDs fixos.


def normalizeText(values: pd.Series) -> pd.Series:
    """Padroniza textos usados na comparação, preservando a fonte.
    Standardizes comparison text while preserving source values.

    Parâmetros / Parameters:
        values: Coluna de texto a padronizar. / Text column to standardize.

    Retorno / Returns:
        Coluna em maiúsculas, sem acentos nas letras latinas nem espaços
        nas extremidades. / Uppercase column without Latin accent marks
        or leading and trailing spaces.
    """
    # ====== Normalização vetorizada ======
    return (
        values.astype('string')
        .str.normalize('NFKD')
        .str.replace(r'[\u0300-\u036f]', '', regex=True)
        .str.strip()
        .str.upper()
    )


def collectMonthlyData(
    session: requests.Session,
    resources: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coleta dados solares de PE e monta o manifesto dos downloads.
    Collects solar data from PE and builds the download manifest.

    Parâmetros / Parameters:
        session: Sessão HTTP. / HTTP session.
        resources: Recursos mensais do catálogo. / Monthly catalog resources.

    Retorno / Returns:
        Base fotovoltaica de PE e manifesto mensal. /
        PE photovoltaic data and monthly manifest.
    """
    monthly_frames = []
    manifest_rows = []

    # ====== Processamento de cada recurso ======
    for number, (month, resource) in enumerate(resources.items(), start=1):
        print(f'[{number:02d}/{len(resources)}] Processando {month}...')
        monthly_data = downloadParquet(session, resource['url'], month)

        # ------ Confere o esquema antes de acessar as colunas ------
        missing_columns = REQUIRED_COLUMNS.difference(monthly_data.columns)
        if missing_columns:
            raise ValueError(
                f'Colunas faltantes no Parquet de {month}: '
                f'{sorted(missing_columns)}'
            )

        # ------ Normaliza apenas os textos usados no filtro ------
        states = normalizeText(monthly_data['id_estado'])
        plant_types = normalizeText(monthly_data['nom_tipousina'])
        mask = states.eq(STATE) & plant_types.eq(PLANT_TYPE)
        filtered_data = monthly_data.loc[mask].copy()
        monthly_frames.append(filtered_data)

        # ------ Registra contagens anteriores ao filtro dos três IDs ------
        manifest_rows.append({
            'mes_arquivo': str(month),
            'nome_recurso': resource['name'],
            'id_recurso': resource['resource_id'],
            'url': resource['url'],
            'linhas_originais': len(monthly_data),
            'linhas_fotovoltaicas_pe': len(filtered_data),
        })
        print(f'  Registros fotovoltaicos em PE: {len(filtered_data):,}')

    # ====== Concatenação do recorte estadual ======
    solar_data = pd.concat(monthly_frames, ignore_index=True)
    if solar_data.empty:
        raise ValueError('Nenhum registro fotovoltaico de PE foi encontrado.')

    return solar_data, pd.DataFrame(manifest_rows)


def prepareSolarData(
    solar_data: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp) -> pd.DataFrame:
    """Aplica os limites temporais e prepara os identificadores do ONS.
    Applies date limits and prepares ONS identifiers.

    Parâmetros / Parameters:
        solar_data: Registros fotovoltaicos de PE. / PE photovoltaic data.
        start: Primeiro instante incluído. / Inclusive start timestamp.
        end: Primeiro instante excluído. / Exclusive end timestamp.

    Retorno / Returns:
        Dados do período com identificadores válidos. /
        Data in the period with valid identifiers.
    """
    # ====== Datas e limites do período ======
    timestamps = pd.to_datetime(solar_data['din_instante'], errors='coerce')
    invalid_dates = timestamps.isna()
    if invalid_dates.any():
        examples = solar_data.loc[invalid_dates, 'din_instante'].head(3)
        raise ValueError(
            f'{invalid_dates.sum()} valor(es) inválido(s) em din_instante. '
            f'Exemplos: {examples.tolist()}'
        )

    mask = timestamps.ge(start) & timestamps.lt(end)
    data = solar_data.loc[mask].copy()
    data['din_instante'] = timestamps.loc[mask]

    # ====== Padronização dos identificadores ======
    data['id_ons'] = data['id_ons'].astype('string').str.strip()
    missing_ids = data['id_ons'].fillna('').eq('')
    if missing_ids.any():
        print(
            '  Registros fora do recorte por falta de id_ons: '
            f'{missing_ids.sum():,}'
        )

    return data.loc[~missing_ids].copy()


# ---------------------------------
# ■■■ COBERTURA E QUALIDADE ■■■
# ---------------------------------
# O relatório estadual permite comparar a cobertura. A verificação
# dos três conjuntos usa a grade esperada para distinguir horas
# ausentes de valores de geração nulos ou iguais a zero.


def summarizeSolarCoverage(data: pd.DataFrame) -> pd.DataFrame:
    """Resume a cobertura observada de cada identificador de PE.
    Summarizes observed coverage for each PE identifier.

    Parâmetros / Parameters:
        data: Registros do período com IDs válidos. /
            Records in the period with valid IDs.

    Retorno / Returns:
        Tabela de contagens e extensão temporal por ID. /
        Counts and time span by ID.
    """
    # ====== Agregação estadual por identificador ======
    indexed_data = data.assign(
        mes_referencia=data['din_instante'].dt.to_period('M'),
    )
    return (
        indexed_data.groupby('id_ons', as_index=False)
        .agg(
            nomes=(
                'nom_usina',
                lambda names: ' | '.join(
                    sorted(names.dropna().astype(str).unique())
                ),
            ),
            primeiro_registro=('din_instante', 'min'),
            ultimo_registro=('din_instante', 'max'),
            meses_com_registro=('mes_referencia', 'nunique'),
            quantidade_registros=('din_instante', 'size'),
            geracao_nao_nula=('val_geracao', 'count'),
        )
        .sort_values(
            ['meses_com_registro', 'quantidade_registros', 'id_ons'],
            ascending=[False, False, True],
            ignore_index=True,
        )
    )


def checkHourlyCoverage(
    data: pd.DataFrame,
    expected_hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Compara os horários dos três conjuntos com a grade esperada.
    Compares timestamps for the three groups against the expected grid.

    Parâmetros / Parameters:
        data: Observações do recorte. / Observations in the selected subset.
        expected_hours: Grade horária do recorte. / Expected hourly grid.

    Retorno / Returns:
        Contagens de horas presentes, ausentes, nulos e duplicatas. /
        Counts of present and missing hours, null values, and duplicates.
    """
    # ====== Comparação com a grade por conjunto ======
    records = []
    for group_id, group in data.groupby('id_ons', sort=True):
        distinct_hours = pd.DatetimeIndex(group['din_instante'].unique())
        hours_on_grid = expected_hours.intersection(distinct_hours)
        duplicate_rows = group['din_instante'].duplicated(keep=False)

        # ------ Conta apenas horários pertencentes à grade ------
        records.append({
            'id_ons': group_id,
            'nome': ' | '.join(sorted(group['nom_usina'].dropna().unique())),
            'horas_esperadas': len(expected_hours),
            'horas_distintas_presentes': len(hours_on_grid),
            'horas_ausentes': len(expected_hours) - len(hours_on_grid),
            'valores_geracao_nulos': group['val_geracao'].isna().sum(),
            'linhas_duplicadas': duplicate_rows.sum(),
        })

    return pd.DataFrame(records)


# ---------------------------------
# ■■■ EXPORTAÇÃO E EXECUÇÃO ■■■
# ---------------------------------
# A execução pode ser feita pelo terminal do VS Code. O módulo também
# pode ser importado para usar as funções sem iniciar a coleta.


def exportResults(
    data: pd.DataFrame,
    coverage: pd.DataFrame,
    verification: pd.DataFrame,
    manifest: pd.DataFrame) -> None:
    """Exporta a base e os relatórios da coleta em dados_ons/.
    Exports the dataset and collection reports to dados_ons/.

    Parâmetros / Parameters:
        data: Base dos três conjuntos. / Dataset for the three groups.
        coverage: Cobertura fotovoltaica de PE. / PE photovoltaic coverage.
        verification: Verificação horária dos três conjuntos. /
            Hourly verification for the three groups.
        manifest: Registro dos recursos mensais. / Monthly resource log.

    Retorno / Returns:
        None. Os resultados são gravados em arquivos. /
        None. Results are written to files.
    """
    # ====== Base consolidada ======
    base_name = 'solar_pe_3_usinas_2024_2025'
    parquet_path = OUTPUT_DIR / f'{base_name}.parquet'
    data.to_parquet(parquet_path, index=False)

    # ====== Base CSV e relatórios ======
    tables = {
        f'{base_name}.csv': data,
        'cobertura_solar_pe_2024_2025.csv': coverage,
        'verificacao_3_usinas_2024_2025.csv': verification,
        'manifesto_downloads_ons.csv': manifest,
    }

    print('\nArquivos gerados:')
    print(f'  {parquet_path}')

    # ------ Ponto decimal para manter a leitura do notebook atual ------
    for file_name, table in tables.items():
        file_path = OUTPUT_DIR / file_name
        table.to_csv(
            file_path,
            index=False,
            sep=';',
            decimal='.',
            encoding='utf-8-sig',
        )
        print(f'  {file_path}')


def main() -> None:
    """Executa a coleta, o recorte e a exportação.
    Runs collection, selection, and export.

    Parâmetros / Parameters:
        Nenhum. / None.

    Retorno / Returns:
        None. As saídas são gravadas em dados_ons/. /
        None. Outputs are written to dados_ons/.
    """
    # ====== Período e recursos mensais ======
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE_EXCLUSIVE)
    months = pd.period_range(start, end - pd.Timedelta(days=1), freq='M')
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        resources = getMonthlyResources(session, months)
        print(f'Arquivos identificados no catálogo: {len(resources)}')
        solar_data, manifest = collectMonthlyData(session, resources)

    # ====== Recorte dos conjuntos e verificação ======
    solar_data = prepareSolarData(solar_data, start, end)
    coverage = summarizeSolarCoverage(solar_data)

    # ------ O recorte usa IDs fixos, mesmo se o ranking estadual mudar ------
    missing_ids = set(SELECTED_IDS).difference(solar_data['id_ons'])
    if missing_ids:
        raise ValueError(
            f'Conjuntos ausentes no recorte: {sorted(missing_ids)}'
        )

    data = (
        solar_data.loc[solar_data['id_ons'].isin(SELECTED_IDS)]
        .sort_values(['id_ons', 'din_instante'], ignore_index=True)
    )

    expected_hours = pd.date_range(start, end, freq='h', inclusive='left')
    verification = checkHourlyCoverage(data, expected_hours)

    # ------ Preserva duplicatas da origem para investigação ------
    if verification['linhas_duplicadas'].gt(0).any():
        print(
            'Atenção: há chaves repetidas. Consulte o relatório.'
        )

    print('\nCobertura das fotovoltaicas de PE:')
    print(coverage.to_string(index=False))
    print('\nVerificação dos três conjuntos:')
    print(verification.to_string(index=False))

    # ====== Gravação dos resultados ======
    exportResults(data, coverage, verification, manifest)


if __name__ == '__main__':
    main()
