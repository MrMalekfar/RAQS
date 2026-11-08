"""Optional study persistence, separated from execution for GUI reuse."""
import json
from pathlib import Path
from ..config.profile_document import profile_document
from .profile_execution import save_profile_results


def save_study_results(directory, study):
    """Write validated outputs under index-based folders, never user-name paths.

    Refuse an existing study manifest rather than mixing two studies. The
    final manifest is written last; a failed export cannot advertise completion.
    """
    from .detection_validation import DetectionValidationResult
    if isinstance(study,DetectionValidationResult):
        return save_detection_validation(directory,study)
    from .detection_study import DetectionStudyResult
    if isinstance(study,DetectionStudyResult):
        return save_detection_results(directory,study)
    from .noise_convergence import ConvergenceResult
    if isinstance(study,ConvergenceResult):
        return save_convergence_results(directory,study)
    root = Path(directory)
    if (root/'study_report.json').exists():
        raise FileExistsError(f'Study output already exists: {root}')
    # Pre-encode before touching files: unsupported metadata must fail visibly.
    report = json.dumps(study.report(), ensure_ascii=False,allow_nan=False,indent=2)
    template = json.dumps(profile_document(study.base_profile),ensure_ascii=False,allow_nan=False,indent=2)
    jobs = []
    if study.nominal_result is not None:
        jobs.append(('nominal',study.nominal_profile,study.nominal_report,study.nominal_result))
    jobs.extend((f'point_{p.index:04d}',p.profile,p.report,p.result) for p in study.points)
    encoded = [(name,json.dumps(profile_document(profile),ensure_ascii=False,allow_nan=False,indent=2),
                json.dumps(run_report,ensure_ascii=False,allow_nan=False,indent=2), result)
               for name,profile,run_report,result in jobs]
    root.mkdir(parents=True,exist_ok=True)
    if (root/'study_profile.json').exists() or any((root/name).exists() for name,_,_,_ in encoded):
        raise FileExistsError(f'Study output contains existing template or point folders: {root}')
    (root/'study_profile.json').write_text(template+'\n',encoding='utf-8')
    for name,profile,run_report,result in encoded:
        sub = root/name
        if sub.exists(): raise FileExistsError(f'Study point output already exists: {sub}')
        sub.mkdir()
        (sub/'profile.json').write_text(profile+'\n',encoding='utf-8')
        (sub/'run_report.json').write_text(run_report+'\n',encoding='utf-8')
        save_profile_results(sub,result)
    (root/'study_report.json').write_text(report+'\n',encoding='utf-8')


def save_detection_results(directory,study):
    """Record calibration and held-out null data as well as every signal trial."""
    root = Path(directory)
    report = json.dumps(study.report(),ensure_ascii=False,allow_nan=False,indent=2)
    template = json.dumps(profile_document(study.base_profile),ensure_ascii=False,allow_nan=False,indent=2)
    refinement_files = {}
    options = study.definition.sensitivity
    if options is not None and options.refinement is not None:
        from ..config.sensitivity import refinement_profile
        from ..config.study import plan_study
        refined = refinement_profile(study.base_profile)
        refinement_files = {'refinement_profile.json':json.dumps(profile_document(refined),ensure_ascii=False,allow_nan=False,indent=2),
                            'refinement_plan.json':json.dumps(plan_study(refined),ensure_ascii=False,allow_nan=False,indent=2)}
    points = [('calibration',study.calibration),('null_validation',study.validation)]
    points += [(f'point_{i:04d}',p) for i,p in enumerate(study.points)]
    encoded = [(name,p,json.dumps(profile_document(p.profile),ensure_ascii=False,allow_nan=False,indent=2),
                json.dumps(p.report,ensure_ascii=False,allow_nan=False,indent=2)) for name,p in points if p is not None]
    if ((root/'study_report.json').exists() or (root/'study_profile.json').exists() or
            any((root/name).exists() for name,_,_,_ in encoded) or
            any((root/name).exists() for name in refinement_files)):
        raise FileExistsError(f'Detection study output already exists: {root}')
    root.mkdir(parents=True,exist_ok=True)
    (root/'study_profile.json').write_text(template+'\n',encoding='utf-8')
    for name,p,profile,run_report in encoded:
        sub = root/name; sub.mkdir()
        (sub/'profile.json').write_text(profile+'\n',encoding='utf-8')
        (sub/'run_report.json').write_text(run_report+'\n',encoding='utf-8')
        save_profile_results(sub,p.result)
    for name,content in refinement_files.items():
        (root/name).write_text(content+'\n',encoding='utf-8')
    (root/'study_report.json').write_text(report+'\n',encoding='utf-8')


def save_detection_validation(directory,study):
    root=Path(directory)
    encoded=json.dumps(study.report(),ensure_ascii=False,allow_nan=False,indent=2)
    template=json.dumps(profile_document(study.base_profile),ensure_ascii=False,allow_nan=False,indent=2)
    names=[f'case_{i:04d}' for i in range(len(study.campaigns))]
    if any((root/name).exists() for name in ['study_report.json','study_profile.json',*names]):
        raise FileExistsError(f'Numerical validation output already exists: {root}')
    root.mkdir(parents=True,exist_ok=True)
    (root/'study_profile.json').write_text(template+'\n',encoding='utf-8')
    for name,(_,campaign) in zip(names,study.campaigns):save_detection_results(root/name,campaign)
    (root/'study_report.json').write_text(encoded+'\n',encoding='utf-8')


def save_convergence_results(directory,study):
    root=Path(directory)
    report=json.dumps(study.report(),ensure_ascii=False,allow_nan=False,indent=2)
    template=json.dumps(profile_document(study.base_profile),ensure_ascii=False,allow_nan=False,indent=2)
    encoded=[(p,json.dumps(profile_document(p.profile),ensure_ascii=False,allow_nan=False,indent=2),
                json.dumps(p.report,ensure_ascii=False,allow_nan=False,indent=2)) for p in study.points]
    if (root/'study_report.json').exists() or (root/'study_profile.json').exists() or any(
            (root/f'point_{p.index:04d}').exists() for p in study.points):
        raise FileExistsError(f'Convergence output already exists: {root}')
    root.mkdir(parents=True,exist_ok=True)
    (root/'study_profile.json').write_text(template+'\n',encoding='utf-8')
    for p,profile,run_report in encoded:
        sub=root/f'point_{p.index:04d}';sub.mkdir()
        (sub/'profile.json').write_text(profile+'\n',encoding='utf-8')
        (sub/'run_report.json').write_text(run_report+'\n',encoding='utf-8')
        save_profile_results(sub,p.result)
    (root/'study_report.json').write_text(report+'\n',encoding='utf-8')
