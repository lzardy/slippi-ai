import time
import hashlib
import os
import zlib
import zipfile
from typing import Any, BinaryIO, NamedTuple, Optional

import boto3
from simplekv.net.boto3store import Boto3Store
from pymongo import MongoClient

from slippi_db.secrets import SECRETS

MB = 10 ** 6

DEFAULTS = dict(
  max_size_per_file=10 * MB,
  min_size_per_file=1 * MB,
  max_files=100,
  max_total_size=250 * MB,
)

# controls where stuff is stored
ENV = os.environ.get('SLIPPI_DB_ENV', 'test')

class S3(NamedTuple):
  session: boto3.Session
  resource: Any
  bucket: Any  # the type here is difficult to specify
  store: Boto3Store

def make_s3() -> S3:
  s3_creds = SECRETS['S3_CREDS']
  access_key, secret_key = s3_creds.split(':')
  session = boto3.Session(access_key, secret_key)
  resource = session.resource('s3')
  bucket = resource.Bucket('slp-replays')
  store = Boto3Store(bucket)
  return S3(session, resource, bucket, store)

s3 = make_s3()

def get_objects(env, stage):
  get_key = lambda path: path.split('/')[2]
  paths = s3.store.iter_keys(prefix=f'{env}/{stage}/')
  return {get_key(path): s3.bucket.Object(path) for path in paths}

client = MongoClient(SECRETS['MONGO_URI'])
db = client.slp_replays

def get_db(env: str, stage: str):
  assert stage in ('raw', 'slp')
  return db.get_collection(env + '-' + stage)

def get_params(env: str) -> dict:
  params_coll = db.params
  found = params_coll.find_one({'env': env})
  if found is None:
    # update params collection
    params = dict(env=env, **DEFAULTS)
    params_coll.insert_one(params)
    return params
  # update found with default params
  for k, v in DEFAULTS.items():
    if k not in found:
      found[k] = v
  return found

def create_params(env: str, **kwargs):
  assert db.params.find_one({'env': env}) is None
  params = dict(env=env, **DEFAULTS)
  params.update(kwargs)
  db.params.insert_one(params)

class Timer:

  def __init__(self, name: str):
    self.name = name
  
  def __enter__(self):
    self.start = time.perf_counter()
  
  def __exit__(self, *_):
    self.duration = time.perf_counter() - self.start
    print(f'{self.name}: {self.duration:.1f}')

def iter_bytes(f, chunk_size=2 ** 16):
  while True:
    chunk = f.read(chunk_size)
    if chunk:
      yield chunk
    else:
      break
  f.seek(0)

class ReplayDB:

  def __init__(self, env: str = ENV):
    self.env = env
    self.params = get_params(env)
    self.raw = db.get_collection(env + '-raw')

  def raw_size(self) -> int:
    total_size = 0
    for doc in self.raw.find():
      total_size += doc['stored_size']
    return total_size

  @property
  def max_file_size(self):
    return self.params['max_size_per_file']

  @property
  def min_file_size(self):
    return self.params['min_size_per_file']

  @property
  def max_files(self):
    return self.params['max_files']

  def max_db_size(self):
    return self.params['max_total_size']

  def upload_slp(self, name: str, content: bytes) -> Optional[str]:
    # max_files = params['max_files']
    # if coll.count_documents({}) >= max_files:
    #   return f'DB full, already have {max_files} uploads.'
    if not name.endswith('.slp'):
      return f'{name}: not a .slp'
    
    max_size = self.params['max_size_per_file']
    if len(content) > max_size:
      return f'{name}: exceeds {max_size} bytes.'
    min_size = self.params['min_size_per_file']
    if len(content) < min_size:
      return f'{name}: must have {min_size} bytes.'

    digest = hashlib.sha256()
    digest.update(content)
    key = digest.hexdigest()

    found = self.raw.find_one({'key': key})
    if found is not None:
      return f'{name}: duplicate file'

    # TODO: validate that file conforms to .slp spec

    # store file in S3
    compressed_bytes = zlib.compress(content)
    s3.store.put(self.name + '.' + key, compressed_bytes)

    # update DB
    self.raw.insert_one(dict(
      key=key,
      name=name,
      type='slp',
      compressed=True,
      original_size=len(content),
      stored_size=len(compressed_bytes),
    ))

    return None

  def upload_zip(self, uploaded):
    errors = []
    with zipfile.ZipFile(uploaded) as zip:
      names = zip.namelist()
      names = [n for n in names if n.endswith('.slp')]
      print(names)

      max_files = self.params['max_files']
      num_uploaded = self.raw.count_documents({})
      if num_uploaded + len(names) > max_files:
        return f'Can\'t upload {len(names)} files, would exceed limit of {max_files}.'

      for name in names:
        with zip.open(name) as f:
          error = self.upload_slp(name, f.read())
          if error:
            errors.append(error)
    
    uploaded.close()
    if errors:
      return '\n'.join(errors)
    return f'Successfully uploaded {len(names)} files.'

  def upload_raw(
    self,
    name: str,
    f: BinaryIO,
    obj_type: Optional[str] = None,
    check_max_size: bool = True,
    **metadata,
  ):
    if obj_type is None:
      obj_type = name.split('.')[-1]

    size = f.seek(0, 2)
    f.seek(0)

    if check_max_size:
      max_bytes_left = self.max_db_size() - self.raw_size()
      if size > max_bytes_left:
        return f'{name}: exceeds {max_bytes_left} bytes'

    with Timer('md5'):
      digest = hashlib.md5()
      for chunk in iter_bytes(f):
        digest.update(chunk)
      key = digest.hexdigest()

    found = self.raw.find_one({'key': key})
    if found is not None:
      return f'{name}: object with md5={key} already uploaded'

    with Timer('upload_fileobj'):
      s3.bucket.upload_fileobj(
          Fileobj=f,
          Key=self.env + '/raw/' + key,
          # ContentLength=size,
          # ContentMD5=str(base64.encodebytes(digest.digest())),
      )
      # store.put_file(self.name + '/raw/' + key, f)

    # update DB
    self.raw.insert_one(dict(
        filename=name,
        key=key,
        hash_method="md5",
        type=obj_type,
        stored_size=size,
        processed=False,
        **metadata,
    ))
    return f'{name}: upload successful'

  def delete(self, key: str):
    s3_key = self.env + '/raw/' + key
    s3.bucket.delete_objects(Delete=dict(Objects=[dict(Key=s3_key)]))
    # store.delete()
    self.raw.delete_one({'key': key})

def nuke_replays(env: str, stage: str):
  db.drop_collection(env + '-' + stage)
  db.params.delete_many({'env': env + '-' + stage})
  keys = s3.store.iter_keys(prefix=f'{env}/{stage}/')
  objects = [dict(Key=k) for k in keys]
  if not objects:
    print('No objects to delete.')
    return
  response = s3.bucket.delete_objects(Delete=dict(Objects=objects))
  print(f'Deleted {len(response["Deleted"])} objects.')